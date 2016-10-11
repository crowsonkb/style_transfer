#!/usr/bin/env python3

"""Neural style transfer using Caffe. Implements A Neural Algorithm of Artistic Style
(http://arxiv.org/abs/1508.06576)."""

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, too-many-locals

from __future__ import division

import argparse
from collections import namedtuple
from fractions import Fraction
import io
import mmap
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
import webbrowser

import numpy as np
from PIL import Image
import posix_ipc
from scipy.ndimage import convolve, convolve1d
import six
from six import print_
from six.moves import cPickle as pickle
from six.moves.BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from six.moves.socketserver import ThreadingMixIn

ARGS = None

if six.PY2:
    CTX = mp
    timer = time.time
else:
    CTX = mp.get_context('spawn')
    timer = time.perf_counter

# Machine epsilon for float32
EPS = np.finfo(np.float32).eps


def normalize(arr):
    """Normalizes an array to have an L1 norm equal to its length."""
    return arr / (np.mean(np.abs(arr)) + EPS)


def resize(arr, size, method=Image.BICUBIC):
    """Resamples a CxHxW Numpy float array to a different HxW shape."""
    if arr.ndim != 3:
        raise TypeError('Only 3D CxHxW arrays are supported')
    h, w = size
    arr = np.float32(arr)
    planes = [arr[i, :, :] for i in range(arr.shape[0])]
    imgs = [Image.fromarray(plane) for plane in planes]
    imgs_resized = [img.resize((w, h), method) for img in imgs]
    return np.stack([np.array(img) for img in imgs_resized])


def roll2(arr, xy):
    """Translates an array by the shift xy, wrapping at the edges."""
    return np.roll(np.roll(arr, xy[0], 2), xy[1], 1)


def gram_matrix(feat):
    """Computes the Gram matrix corresponding to a feature map."""
    n, mh, mw = feat.shape
    feat = feat.reshape((n, mh * mw))
    gram = np.dot(feat, feat.T) / np.float32(feat.size)
    return gram


# pylint: disable=no-member
class SharedNDArray:
    """Creates an ndarray shared between processes using POSIX shared memory. It can be used to
    transmit ndarrays between processes quickly. It can be sent over multiprocessing.Pipe and
    Queue."""
    def __init__(self, shape, dtype=np.float64, name=None):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if name:
            self._shm = posix_ipc.SharedMemory(name)
        else:
            self._shm = posix_ipc.SharedMemory(None, posix_ipc.O_CREX, size=size)
        buf = mmap.mmap(self._shm.fd, size)
        self.array = np.ndarray(shape, dtype, buf)

    @classmethod
    def copy(cls, arr):
        """Creates a new SharedNDArray that is a copy of the given ndarray."""
        new_shm = cls.zeros_like(arr)
        new_shm.array[:] = arr
        return new_shm

    @classmethod
    def zeros_like(cls, arr):
        """Creates a new zero-filled SharedNDArray with the shape and dtype of the given
        ndarray."""
        return cls(arr.shape, arr.dtype)

    def unlink(self):
        """Marks the ndarray for deletion. This method should be called once and only once, from
        one process."""
        self._shm.unlink()

    def __del__(self):
        self._shm.close_fd()

    def __getstate__(self):
        return self.array.shape, self.array.dtype, self._shm.name

    def __setstate__(self, state):
        self.__init__(*state)


class LayerIndexer:
    """Helper class for accessing feature maps and gradients."""
    def __init__(self, net, attr):
        self.net, self.attr = net, attr

    def __getitem__(self, key):
        return getattr(self.net.blobs[key], self.attr)[0]

    def __setitem__(self, key, value):
        getattr(self.net.blobs[key], self.attr)[0] = value


class Optimizer:
    """Implements the Adam gradient descent optimizer with Polyak-Ruppert averaging."""
    def __init__(self, params, step_size=1, averaging=True, b1=0.9, b2=0.999):
        """Initializes the optimizer."""
        self.params = params
        self.step_size = step_size
        self.averaging = averaging
        self.b1 = b1
        self.b2 = b2
        self.step = 0
        self.xy = np.zeros(2, dtype=np.int32)
        self.g1 = np.zeros_like(params)
        self.g2 = np.zeros_like(params)
        self.p1 = params.copy()

    def update(self, grad):
        """Returns a step's parameter update given its gradient."""
        self.step += 1

        # Adam
        self.g1[:] = self.b1*self.g1 + (1-self.b1)*grad
        self.g2[:] = self.b2*self.g2 + (1-self.b2)*grad**2
        g1_bar = self.b1*self.g1 + (1-self.b1)*grad
        g1_hat = g1_bar/(1-self.b1**(self.step+1))
        g2_hat = self.g2/(1-self.b2**self.step)
        self.params -= self.step_size * g1_hat / (np.sqrt(g2_hat) + EPS)

        # Polyak-Ruppert averaging
        weight = 1 / self.step
        self.p1[:] = (1-weight)*self.p1 + weight*self.params
        if self.averaging:
            return self.p1
        else:
            return self.params

    def roll(self, xy):
        """Rolls the optimizer's internal state."""
        self.xy += xy
        self.g1[:] = roll2(self.g1, xy)
        self.g2[:] = roll2(self.g2, xy)
        self.p1[:] = roll2(self.p1, xy)

    def set_params(self, last_iterate):
        """Sets params to the supplied array (a possibly-resized or altered last non-averaged
        iterate), resampling the optimizer's internal state if the shape has changed."""
        # P-R averaging should only average over the current scale. For some reason the result
        # looks better if Adam is provided with an incorrect step number for its resampled internal
        # state.
        self.step = 0
        self.params = last_iterate
        hw = self.params.shape[-2:]
        self.g1 = resize(self.g1, hw)
        self.g2 = resize(self.g2, hw, Image.NEAREST)
        self.p1 = resize(self.p1, hw)

    def restore_state(self, optimizer):
        """Given an Optimizer instance, restores internal state from it."""
        self.params = optimizer.params
        self.g1 = optimizer.g1
        self.g2 = optimizer.g2
        self.p1 = optimizer.p1
        self.step = optimizer.step
        self.xy = optimizer.xy.copy()
        self.roll(-self.xy)

FeatureMapRequest = namedtuple('FeatureMapRequest', 'resp img layers')
FeatureMapResponse = namedtuple('FeatureMapResponse', 'resp features')
SCGradRequest = namedtuple(
    'SCGradRequest', 'resp img roll start content_layers style_layers content_weight style_weight')
SCGradResponse = namedtuple('SCGradResponse', 'resp grad')
SetFeaturesAndGrams = namedtuple('SetFeaturesAndGrams', 'features grams')


class TileWorker:
    """Computes feature maps and gradients on the specified device in a separate process."""
    def __init__(self, req_q, resp_q, model, device=-1):
        self.req_q = req_q
        self.resp_q = resp_q
        self.model = None
        self.model_info = (model.deploy, model.weights, model.mean, model.bgr)
        self.device = device
        self.features = None
        self.grams = None
        self.proc = CTX.Process(target=self.run)
        self.proc.daemon = True
        self.proc.start()

    def __del__(self):
        if not self.proc.exitcode:
            self.proc.terminate()

    def run(self):
        """This method runs in the new process."""
        if self.device >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device)
        import caffe
        if self.device >= 0:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        caffe.set_random_seed(0)
        np.random.seed(0)

        self.model = CaffeModel(*self.model_info)
        self.model.img = np.zeros((3, 1, 1), dtype=np.float32)

        while True:
            req = self.req_q.get()
            layers = []

            if isinstance(req, FeatureMapRequest):
                for layer in reversed(self.model.layers()):
                    if layer in req.layers:
                        layers.append(layer)
                features = self.model.eval_features_tile(req.img.array, layers)
                req.img.unlink()
                features_shm = {layer: SharedNDArray.copy(features[layer]) for layer in features}
                self.resp_q.put(FeatureMapResponse(req.resp, features_shm))

            if isinstance(req, SCGradRequest):
                for layer in reversed(self.model.layers()):
                    if layer in req.content_layers or layer in req.style_layers:
                        layers.append(layer)
                self.model.roll(req.roll, jitter_scale=1)
                grad = self.model.eval_sc_grad_tile(
                    req.img.array, req.start, layers, req.content_layers, req.style_layers,
                    req.content_weight, req.style_weight)
                req.img.unlink()
                self.model.roll(-req.roll, jitter_scale=1)
                self.resp_q.put(SCGradResponse(req.resp, SharedNDArray.copy(grad)))

            if isinstance(req, SetFeaturesAndGrams):
                self.model.features = \
                    {layer: req.features[layer].array.copy() for layer in req.features}
                self.model.grams = \
                    {layer: req.grams[layer].array.copy() for layer in req.grams}
                self.resp_q.put(())


class TileWorkerPoolError(Exception):
    """Indicates abnormal termination of TileWorker processes."""
    pass


class TileWorkerPool:
    """A collection of TileWorkers."""
    def __init__(self, model, devices):
        self.workers = []
        self.next_worker = 0
        self.resp_q = CTX.Queue()
        self.is_healthy = True
        for device in devices:
            self.workers.append(
                TileWorker(CTX.Queue(), self.resp_q, model, device))

    def __del__(self):
        self.is_healthy = False
        for worker in self.workers:
            worker.__del__()

    def request(self, req):
        """Enqueues a request."""
        self.workers[self.next_worker].req_q.put(req)
        self.next_worker = (self.next_worker + 1) % len(self.workers)

    def reset_next_worker(self):
        """Sets the worker which will process the next request to worker 0."""
        self.next_worker = 0

    def ensure_healthy(self):
        """Checks for abnormal pool process termination."""
        if not self.is_healthy:
            raise TileWorkerPoolError('Workers already terminated')
        for worker in self.workers:
            if worker.proc.exitcode:
                self.__del__()
                raise TileWorkerPoolError('Pool malfunction; terminating')

    def set_features_and_grams(self, features, grams):
        """Propagates feature maps and Gram matrices to all TileWorkers."""
        for worker in self.workers:
            features_shm = {layer: SharedNDArray.copy(features[layer]) for layer in features}
            grams_shm = {layer: SharedNDArray.copy(grams[layer]) for layer in grams}
            worker.req_q.put(SetFeaturesAndGrams(features_shm, grams_shm))
        for _ in self.workers:
            self.resp_q.get()
        _ = [shm.unlink() for shm in features_shm.values()]
        _ = [shm.unlink() for shm in grams_shm.values()]


class CaffeModel:
    """A Caffe neural network model."""
    def __init__(self, deploy, weights, mean=(0, 0, 0), bgr=True):
        import caffe
        self.deploy = deploy
        self.weights = weights
        self.mean = np.float32(mean).reshape((3, 1, 1))
        self.bgr = bgr
        self.last_layer = 'pool5'
        self.net = caffe.Net(self.deploy, 1, weights=self.weights)
        self.data = LayerIndexer(self.net, 'data')
        self.diff = LayerIndexer(self.net, 'diff')
        self.features = None
        self.grams = None
        self.img = None

    def get_image(self, params=None):
        """Gets the current model input (or provided alternate input) as a PIL image."""
        if params is None:
            params = self.img
        arr = params + self.mean
        if self.bgr:
            arr = arr[::-1]
        arr = arr.transpose((1, 2, 0))
        return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))

    def set_image(self, img):
        """Sets the current model input to a PIL image."""
        arr = np.float32(img).transpose((2, 0, 1))
        if self.bgr:
            arr = arr[::-1]
        self.img = arr - self.mean

    def layers(self):
        """Returns the layer names of the network."""
        layers = []
        for i, layer in enumerate(self.net.blobs.keys()):
            if i == 0:
                continue
            if layer.find('_split_') == -1:
                layers.append(layer)
        return layers

    @staticmethod
    def layer_info(layer):
        """Returns the number of channels and the scale factor vs. the image for a VGG layer."""
        assert layer.startswith('conv') or layer.startswith('pool')
        level = int(layer[4])-1
        channels = (64, 128, 256, 512, 512)[level]
        if layer.startswith('pool'):
            level += 1
        return 2**level, channels

    def eval_features_tile(self, img, layers):
        """Computes a single tile in a set of feature maps."""
        self.net.blobs['data'].reshape(1, 3, *img.shape[-2:])
        self.data['data'] = img
        self.net.forward(end=self.last_layer)
        return {layer: self.data[layer] for layer in layers}

    def eval_features_once(self, pool, layers, tile_size=512):
        """Computes the set of feature maps for an image."""
        img_size = np.array(self.img.shape[-2:])
        ntiles = (img_size-1) // tile_size + 1
        tile_size = img_size // ntiles
        print_('Using %dx%d tiles of size %dx%d.' %
               (ntiles[1], ntiles[0], tile_size[1], tile_size[0]))
        features = {}
        for layer in layers:
            scale, channels = self.layer_info(layer)
            shape = (channels,) + tuple(np.int32(np.ceil(img_size / scale)))
            features[layer] = np.zeros(shape, dtype=np.float32)
        for y in range(ntiles[0]):
            for x in range(ntiles[1]):
                xy = np.array([y, x])
                start = xy * tile_size
                end = start + tile_size
                if y == ntiles[0] - 1:
                    end[0] = img_size[0]
                if x == ntiles[1] - 1:
                    end[1] = img_size[1]
                tile = self.img[:, start[0]:end[0], start[1]:end[1]]
                pool.ensure_healthy()
                pool.request(FeatureMapRequest(start, SharedNDArray.copy(tile), layers))
        pool.reset_next_worker()
        for _ in range(np.prod(ntiles)):
            start, feats_tile = pool.resp_q.get()
            for layer, feat in feats_tile.items():
                scale, _ = self.layer_info(layer)
                start_f = start // scale
                end_f = start_f + np.array(feat.array.shape[-2:])
                features[layer][:, start_f[0]:end_f[0], start_f[1]:end_f[1]] = feat.array
                feat.unlink()

        return features

    def prepare_features(self, pool, layers, tile_size=512, passes=10):
        """Averages the set of feature maps for an image over multiple passes to obscure tiling."""
        img_size = np.array(self.img.shape[-2:])
        if max(*img_size) <= tile_size:
            passes = 1
        self.features = {}
        for i in range(passes):
            xy = np.array((0, 0))
            if i > 0:
                xy = np.int32(np.random.uniform(size=2) * img_size) // 32
            self.roll(xy)
            feats = self.eval_features_once(pool, layers, tile_size)
            if not self.features:
                self.features = feats
            else:
                for layer in self.features:
                    self.features[layer] += feats[layer]
            self.roll(-xy)
        for layer in self.features:
            self.features[layer] /= passes
        return self.features

    def preprocess_images(self, pool, content_image, style_image, content_layers, style_layers,
                          tile_size=512):
        """Performs preprocessing tasks on the input images."""
        # Construct list of layers to visit during the backward pass
        layers = []
        for layer in reversed(self.layers()):
            if layer in content_layers or layer in style_layers:
                layers.append(layer)

        # Prepare Gram matrices from style image
        print_('Preprocessing the style image...')
        self.grams = {}
        self.set_image(style_image)
        feats = self.prepare_features(pool, style_layers, tile_size)
        for layer in feats:
            self.grams[layer] = gram_matrix(feats[layer])

        # Prepare feature maps from content image
        print_('Preprocessing the content image...')
        self.set_image(content_image)
        self.prepare_features(pool, content_layers, tile_size)

        return layers

    def eval_sc_grad_tile(self, img, start, layers, content_layers, style_layers,
                          content_weight, style_weight):
        """Evaluates an individual style+content gradient tile."""
        self.net.blobs['data'].reshape(1, 3, *img.shape[-2:])
        self.data['data'] = img

        # Prepare gradient buffers and run the model forward
        for layer in layers:
            self.diff[layer] = 0
        self.net.forward(end=self.last_layer)
        # content_loss, style_loss = 0, 0

        for i, layer in enumerate(layers):
            # Compute the content and style gradients
            if layer in content_layers:
                scale, _ = self.layer_info(layer)
                start = start // scale
                end = start + np.array(self.data[layer].shape[-2:])
                feat = self.features[layer][:, start[0]:end[0], start[1]:end[1]]
                c_grad = self.data[layer] - feat
                self.diff[layer] += normalize(c_grad)*content_weight
                # content_loss += 0.5 * np.sum((self.data[layer] - self.features[layer])**2)
            if layer in style_layers:
                current_gram = gram_matrix(self.data[layer])
                n, mh, mw = self.data[layer].shape
                feat = self.data[layer].reshape((n, mh * mw))
                s_grad = np.dot(current_gram - self.grams[layer], feat)
                s_grad = s_grad.reshape((n, mh, mw))
                self.diff[layer] += normalize(s_grad)*style_weight
                # style_loss += 0.5 * np.sum((current_gram - self.grams[layer])**2)

            # Run the model backward
            if i+1 == len(layers):
                self.net.backward(start=layer)
            else:
                self.net.backward(start=layer, end=layers[i+1])

        return self.diff['data']

    def eval_sc_grad(self, pool, roll, content_layers, style_layers, content_weight, style_weight,
                     tile_size=512):
        """Evaluates the summed style and content gradients."""
        grad = np.zeros_like(self.img)
        img_size = np.array(self.img.shape[-2:])
        ntiles = (img_size-1) // tile_size + 1
        tile_size = img_size // ntiles

        for y in range(ntiles[0]):
            for x in range(ntiles[1]):
                xy = np.array([y, x])
                start = xy * tile_size
                end = start + tile_size
                if y == ntiles[0] - 1:
                    end[0] = img_size[0]
                if x == ntiles[1] - 1:
                    end[1] = img_size[1]
                tile = self.img[:, start[0]:end[0], start[1]:end[1]]
                pool.ensure_healthy()
                pool.request(
                    SCGradRequest((start, end), SharedNDArray.copy(tile), roll, start,
                                  content_layers, style_layers, content_weight, style_weight))
        pool.reset_next_worker()
        for _ in range(np.prod(ntiles)):
            (start, end), grad_tile = pool.resp_q.get()
            grad[:, start[0]:end[0], start[1]:end[1]] = grad_tile.array

        return grad

    def roll(self, xy, jitter_scale=32):
        """Rolls image and feature maps."""
        xy = xy * jitter_scale
        for layer, feat in self.features.items():
            scale, _ = self.layer_info(layer)
            self.features[layer][:] = roll2(feat, xy // scale)
        self.img[:] = roll2(self.img, xy)


class StyleTransfer:
    """Performs style transfer."""
    def __init__(self, model):
        self.model = model
        self.current_output = None
        self.optimizer = None
        self.pool = None
        self.step = 0

    def transfer(self, iterations, params, content_image, style_image, callback=None):
        """Performs style transfer from style_image to content_image."""
        content_weight = ARGS.content_weight / max(len(ARGS.content_layers), 1)
        style_weight = 1 / max(len(ARGS.style_layers), 1)

        layers = self.model.preprocess_images(
            self.pool, content_image, style_image, ARGS.content_layers, ARGS.style_layers,
            ARGS.tile_size)
        self.pool.set_features_and_grams(self.model.features, self.model.grams)
        self.model.img = params

        old_img = self.optimizer.p1.copy()
        self.step += 1
        log = open('log.csv', 'w')
        print_('tv loss', file=log, flush=True)

        for step in range(1, iterations+1):
            # Forward jitter
            jitter_scale, _ = self.model.layer_info(
                [l for l in layers if l in ARGS.content_layers][0])
            xy = np.array((0, 0))
            img_size = np.array(self.model.img.shape[-2:])
            if max(*img_size) > ARGS.tile_size:
                xy = np.int32(np.random.uniform(-0.5, 0.5, size=2) * img_size) // jitter_scale
            self.model.roll(xy, jitter_scale=jitter_scale)
            self.optimizer.roll(xy * jitter_scale)

            # Compute style+content gradient
            grad = self.model.eval_sc_grad(self.pool, xy * jitter_scale, ARGS.content_layers,
                                           ARGS.style_layers, content_weight, style_weight,
                                           tile_size=ARGS.tile_size)

            # Compute total variation gradient
            tv_kernel = np.float32([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]])
            tv_grad = convolve(self.model.img, tv_kernel, mode='wrap')/255

            # Selectively blur edges more to obscure jitter and tile seams
            tv_mask = np.ones_like(tv_grad)
            tv_mask[:, :2, :] = 5
            tv_mask[:, -2:, :] = 5
            tv_mask[:, :, :2] = 5
            tv_mask[:, :, -2:] = 5
            tv_grad *= tv_mask

            # Compute a weighted sum of normalized gradients
            grad = normalize(grad) + ARGS.tv_weight*tv_grad

            # In-place gradient descent update
            avg_img = self.optimizer.update(grad)

            # Backward jitter
            self.model.roll(-xy, jitter_scale=jitter_scale)
            self.optimizer.roll(-xy * jitter_scale)

            # Apply constraints
            mean = self.model.mean.squeeze()
            self.model.img[0] = np.clip(self.model.img[0], -mean[0], 255-mean[0])
            self.model.img[1] = np.clip(self.model.img[1], -mean[1], 255-mean[1])
            self.model.img[2] = np.clip(self.model.img[2], -mean[2], 255-mean[2])

            # Compute update size statistic
            update_size = np.mean(np.abs(avg_img - old_img))
            old_img[:] = avg_img

            # Compute tv loss statistic
            tv_h = convolve1d(avg_img, [-1, 1], axis=1, mode='wrap')
            tv_v = convolve1d(avg_img, [-1, 1], axis=2, mode='wrap')
            tv_loss = 0.5 * np.sum(tv_h**2 + tv_v**2) / avg_img.size
            print_(tv_loss, file=log, flush=True)

            self.current_output = self.model.get_image(avg_img)

            if callback is not None:
                callback(step=step, update_size=update_size, loss=tv_loss)

        return self.current_output, self.model.get_image()

    def transfer_multiscale(self, sizes, iterations, content_image, style_image, initial_image,
                            initial_state=None, **kwargs):
        """Performs style transfer from style_image to content_image at the given sizes."""
        output_image = None
        last_iterate = None
        print_('Starting %d worker process(es).' % len(ARGS.devices))
        self.pool = TileWorkerPool(self.model, ARGS.devices)

        for i, size in enumerate(sizes):
            content_scaled = resize_to_fit(content_image, size, scale_up=True)
            style_scaled = resize_to_fit(style_image, round(size * ARGS.style_scale))
            if output_image:  # this is not the first scale
                initial_image = last_iterate.resize(content_scaled.size, Image.BICUBIC)
                self.model.set_image(initial_image)
                params = self.model.img
                self.optimizer.set_params(params)
            else:  # this is the first scale
                if initial_image:  # and the user supplied an initial image
                    initial_image = initial_image.resize(content_scaled.size, Image.BICUBIC)
                    self.model.set_image(initial_image)
                else:  # and the user did not supply an initial image
                    w, h = content_scaled.size
                    self.model.set_image(np.random.uniform(0, 255, size=(h, w, 3)))

                # make sure the optimizer's params array shares memory with self.model.img
                # after preprocess_image is called later
                self.optimizer = Optimizer(
                    self.model.img, step_size=ARGS.step_size, averaging=not ARGS.no_averaging)

                if initial_state:
                    self.optimizer.restore_state(initial_state)
                    if self.model.img.shape != self.optimizer.params.shape:
                        initial_image = self.model.get_image(self.optimizer.params)
                        initial_image = initial_image.resize(content_scaled.size, Image.BICUBIC)
                        self.model.set_image(initial_image)
                        self.optimizer.set_params(self.model.img)
                    self.model.img = self.optimizer.params

            params = self.model.img
            iters_i = iterations[min(i, len(iterations)-1)]
            output_image, last_iterate = self.transfer(iters_i, params, content_scaled,
                                                       style_scaled, **kwargs)

        return output_image

    def save_state(self, filename='out.state'):
        """Saves the optimizer's internal state to disk."""
        with open(filename, 'wb') as f:
            pickle.dump(self.optimizer, f, pickle.HIGHEST_PROTOCOL)


class Progress:
    """A helper class for keeping track of progress."""
    prev_t = None
    t = 0
    step = 0
    update_size = np.nan
    loss = np.nan

    def __init__(self, transfer, url=None, steps=-1, save_every=0):
        self.transfer = transfer
        self.url = url
        self.steps = steps
        self.save_every = save_every

    def __call__(self, step=-1, update_size=np.nan, loss=np.nan):
        this_t = timer()
        self.step += 1
        self.update_size = update_size
        self.loss = loss
        if self.save_every and self.step % self.save_every == 0:
            self.transfer.current_output.save('out_%04d.png' % self.step)
        if self.step == 1:
            if self.url:
                webbrowser.open(self.url)
        else:
            self.t = this_t - self.prev_t
        print_('Step %d, time: %.2f s, mean update: %.2f, mean tv loss: %.1f' %
               (step, self.t, update_size, loss), flush=True)
        self.prev_t = this_t


class ProgressServer(ThreadingMixIn, HTTPServer):
    """HTTP server class."""
    transfer = None
    progress = None
    hidpi = False


class ProgressHandler(BaseHTTPRequestHandler):
    """Serves intermediate outputs over HTTP."""
    index = """
    <meta http-equiv="refresh" content="5">
    <style>
    body {
        background-color: rgb(55, 55, 55);
        color: rgb(255, 255, 255);
    }
    #out {image-rendering: -webkit-optimize-contrast;}</style>
    <h1>Style transfer</h1>
    <img src="/out.png" id="out" width="%(w)d" height="%(h)d">
    <p>Step %(step)d/%(steps)d, time: %(t).2f s/step, mean update: %(update_size).2f,
    mean tv loss: %(loss).1f
    """

    def do_GET(self):
        """Retrieves index.html or an intermediate output."""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            scale = 1
            if self.server.hidpi:
                scale = 2
            self.wfile.write((self.index % {
                'step': self.server.progress.step,
                'steps': self.server.progress.steps,
                't': self.server.progress.t,
                'update_size': self.server.progress.update_size,
                'loss': self.server.progress.loss,
                'w': self.server.transfer.current_output.size[0] / scale,
                'h': self.server.transfer.current_output.size[1] / scale,
            }).encode())
        elif self.path == '/out.png':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            buf = io.BytesIO()
            self.server.transfer.current_output.save(buf, format='png')
            self.wfile.write(buf.getvalue())
        else:
            self.send_error(404)


def resize_to_fit(image, size, scale_up=False):
    """Resizes image to fit into a size-by-size square."""
    size = int(round(size))
    w, h = image.size
    if not scale_up and max(w, h) <= size:
        return image
    new_w, new_h = w, h
    if w > h:
        new_w = size
        new_h = int(round(size * h/w))
    else:
        new_h = size
        new_w = int(round(size * w/h))
    return image.resize((new_w, new_h), Image.LANCZOS)


def ffloat(s):
    """Parses fractional or floating point input strings."""
    return float(Fraction(s))


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('content_image', help='the content image')
    parser.add_argument('style_image', help='the style image')
    parser.add_argument('output_image', nargs='?', default='out.png', help='the output image')
    parser.add_argument('--init', metavar='IMAGE', help='the initial image')
    parser.add_argument('--state', help='a .state file (the initial state)')
    parser.add_argument(
        '--iterations', '-i', nargs='+', type=int, default=[300], help='the number of iterations')
    parser.add_argument(
        '--step-size', '-st', type=ffloat, default=15, help='the step size (iteration magnitude)')
    parser.add_argument(
        '--size', '-s', nargs='+', type=int, default=[256], help='the output size(s)')
    parser.add_argument(
        '--style-scale', '-ss', type=ffloat, default=1, help='the style scale factor')
    parser.add_argument(
        '--content-weight', '-cw', type=ffloat, default=0.05, help='the content image factor')
    parser.add_argument(
        '--tv-weight', '-tw', type=ffloat, default=1, help='the smoothing factor')
    parser.add_argument(
        '--no-averaging', default=False, action='store_true',
        help='disable averaging of successive iterates')
    parser.add_argument(
        '--content-layers', nargs='*', default=['conv4_2'],
        metavar='LAYER', help='the layers to use for content')
    parser.add_argument(
        '--style-layers', nargs='*', metavar='LAYER',
        default=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        help='the layers to use for style')
    parser.add_argument(
        '--port', '-p', type=int, default=8000,
        help='the port to use for the http server')
    parser.add_argument(
        '--no-browser', action='store_true', help='don\'t open a web browser')
    parser.add_argument(
        '--hidpi', action='store_true', help='display the image at 2x scale in the browser')
    parser.add_argument(
        '--model', default='VGG_ILSVRC_19_layers_deploy.prototxt',
        help='the Caffe deploy.prototxt for the model to use')
    parser.add_argument(
        '--weights', default='VGG_ILSVRC_19_layers.caffemodel',
        help='the Caffe .caffemodel for the model to use')
    parser.add_argument(
        '--mean', nargs=3, metavar=('B_MEAN', 'G_MEAN', 'R_MEAN'),
        default=(103.939, 116.779, 123.68),
        help='the per-channel means of the model (BGR order)')
    parser.add_argument(
        '--list-layers', action='store_true', help='list the model\'s layers')
    parser.add_argument(
        '--save-every', metavar='N', type=int, default=0, help='save the image every n steps')
    parser.add_argument(
        '--devices', nargs='+', metavar='DEVICE', type=int, default=[0],
        help='device numbers to use (-1 for cpu)')
    parser.add_argument(
        '--tile-size', type=int, default=512, help='the maximum rendering tile size')
    parser.add_argument(
        '--seed', type=int, default=0, help='the random seed')
    global ARGS  # pylint: disable=global-statement
    ARGS = parser.parse_args()


def main():
    """CLI interface for style transfer."""
    start_time = timer()
    parse_args()

    os.environ['GLOG_minloglevel'] = '2'
    import caffe
    caffe.set_mode_cpu()

    print_('Loading %s.' % ARGS.weights)
    model = CaffeModel(ARGS.model, ARGS.weights, ARGS.mean)
    transfer = StyleTransfer(model)
    if ARGS.list_layers:
        print_('Layers:')
        for layer in model.layers():
            print_('    %s, size=%s' % (layer, model.data[layer].shape))
        sys.exit(0)

    sizes = sorted(ARGS.size)
    content_image = Image.open(ARGS.content_image).convert('RGB')
    style_image = Image.open(ARGS.style_image).convert('RGB')
    initial_image = None
    if ARGS.init:
        initial_image = Image.open(ARGS.init).convert('RGB')

    server_address = ('', ARGS.port)
    url = 'http://127.0.0.1:%d/' % ARGS.port
    server = ProgressServer(server_address, ProgressHandler)
    server.transfer = transfer
    server.hidpi = ARGS.hidpi
    progress_args = {}
    if not ARGS.no_browser:
        progress_args['url'] = url
    steps = 0
    for i in range(len(sizes)):
        steps += ARGS.iterations[min(i, len(ARGS.iterations)-1)]
    server.progress = Progress(
        transfer, steps=steps, save_every=ARGS.save_every, **progress_args)
    th = threading.Thread(target=server.serve_forever)
    th.daemon = True
    th.start()
    print_('\nWatch the progress at: %s\n' % url)

    state = None
    if ARGS.state:
        state = pickle.load(open(ARGS.state, 'rb'))

    np.random.seed(ARGS.seed)
    try:
        transfer.transfer_multiscale(
            sizes, ARGS.iterations, content_image, style_image, initial_image,
            callback=server.progress, initial_state=state)
    except KeyboardInterrupt:
        print_()

    if transfer.current_output:
        print_('Saving output as %s.' % ARGS.output_image)
        transfer.current_output.save(ARGS.output_image)
        a, _, _ = ARGS.output_image.rpartition('.')
        print_('Saving state as %s.' % (a + '.state'))
        transfer.save_state(a + '.state')
    time_spent = timer() - start_time
    print_('Exiting after %dm %.2fs.' % (time_spent // 60, time_spent % 60))

if __name__ == '__main__':
    main()
