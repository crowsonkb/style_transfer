#!/usr/bin/env python3

"""Neural style transfer using Caffe. Implements A Neural Algorithm of Artistic Style
(http://arxiv.org/abs/1508.06576)."""

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, too-many-locals

from __future__ import division

import argparse
import configparser
from collections import namedtuple, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import partial
import io
import json
import mmap
import multiprocessing as mp
import os
import pickle
import shlex
import sys
import threading
import time
import webbrowser

import numpy as np
from PIL import Image, PngImagePlugin
import posix_ipc
from scipy.linalg import blas
from scipy.ndimage import zoom
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
    CTX = mp.get_context('fork')
    timer = time.perf_counter

# Machine epsilon for float32
EPS = np.finfo(np.float32).eps

# Maximum number of MKL threads between all processes
MKL_THREADS = None


def set_thread_count(threads):
    """Sets the maximum number of MKL threads for this process."""
    if MKL_THREADS is not None:
        mkl.set_num_threads(max(1, threads))

try:
    import mkl
    MKL_THREADS = mkl.get_max_threads()
    set_thread_count(1)
except ImportError:
    pass


# pylint: disable=no-member
def dot(x, y):
    """Returns the dot product of two float32 arrays with the same shape."""
    return blas.sdot(x.ravel(), y.ravel())


# pylint: disable=no-member
def axpy(a, x, y):
    """Sets y = a*x + y for float a and float32 arrays x, y and returns y."""
    y_ = blas.saxpy(x.ravel(), y.ravel(), a=a).reshape(y.shape)
    if y is not y_:
        y[:] = y_
    return y


def norm2(arr):
    """Returns 1/2 the L2 norm squared."""
    return dot(arr, arr) / 2


def normalize(arr):
    """Normalizes an array in-place to have an L1 norm equal to its size."""
    arr /= np.mean(abs(arr)) + EPS
    return arr


def resize(a, hw, method=Image.LANCZOS):
    """Resamples an image array in CxHxW format to a new HxW size. The interpolation is performed
    in floating point and the result dtype is numpy.float32."""
    def _resize(a, b):
        b[:] = Image.fromarray(a).resize((hw[1], hw[0]), method)

    a = np.float32(a)
    ch = a.shape[0]
    b = np.zeros((ch, hw[0], hw[1]), np.float32)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = [ex.submit(_resize, a[i], b[i]) for i in range(ch)]
        _ = [fut.result() for fut in futs]

    return b


def roll_by_1(arr, shift, axis):
    """Rolls a 3D array in-place by a shift of one element. Axes 1 and 2 only."""
    if axis == 1:
        if shift == -1:
            line = arr[:, 0, :].copy()
            arr[:, :-1, :] = arr[:, 1:, :]
            arr[:, -1, :] = line
        elif shift == 1:
            line = arr[:, -1, :].copy()
            arr[:, 1:, :] = arr[:, :-1, :]
            arr[:, 0, :] = line
    elif axis == 2:
        if shift == -1:
            line = arr[:, :, 0].copy()
            arr[:, :, :-1] = arr[:, :, 1:]
            arr[:, :, -1] = line
        elif shift == 1:
            line = arr[:, :, -1].copy()
            arr[:, :, 1:] = arr[:, :, :-1]
            arr[:, :, 0] = line
    else:
        raise ValueError('Unsupported shift or axis')
    return arr


def roll2(arr, xy):
    """Translates an array by the shift xy, wrapping at the edges."""
    if (xy == 0).all():
        return arr
    # return np.roll(arr, xy, (2, 1))
    return np.roll(np.roll(arr, xy[0], -1), xy[1], -2)


def gram_matrix(feat):
    """Computes the Gram matrix corresponding to a feature map."""
    n, mh, mw = feat.shape
    feat = feat.reshape((n, mh * mw))
    return blas.ssyrk(1 / feat.size, feat)


def tv_norm(x, beta=2):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis and [3]."""
    x_diff = x - roll_by_1(x.copy(), -1, axis=2)
    y_diff = x - roll_by_1(x.copy(), -1, axis=1)
    grad_norm2 = x_diff**2 + y_diff**2 + EPS
    loss = np.sum(grad_norm2**(beta/2))
    dgrad_norm = (beta/2) * grad_norm2**(beta/2 - 1)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad -= roll_by_1(dx_diff, 1, axis=2)
    grad -= roll_by_1(dy_diff, 1, axis=1)
    return loss, grad


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
        self._buf = mmap.mmap(self._shm.fd, size)
        self.array = np.ndarray(shape, dtype, self._buf)

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
        self._buf.close()
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


class AdamOptimizer:
    """Implements the Adam gradient descent optimizer [4] with iterate averaging."""
    def __init__(self, params, step_size=1, b1=0.9, b2=0.999, bp1=0):
        """Initializes the optimizer."""
        self.params = params
        self.step_size = step_size
        self.b1, self.b2, self.bp1 = b1, b2, bp1

        self.step = 0
        self.xy = np.zeros(2, dtype=np.int32)
        self.g1 = np.zeros_like(params)
        self.g2 = np.zeros_like(params)
        self.p1 = np.zeros_like(params)

    def update(self, opfunc):
        """Returns a step's parameter update given a loss/gradient evaluation function."""
        self.step += 1
        loss, grad = opfunc(self.params)

        # Adam
        self.g1 *= self.b1
        axpy(1 - self.b1, grad, self.g1)
        self.g2 *= self.b2
        axpy(1 - self.b2, grad**2, self.g2)
        step_size = self.step_size * np.sqrt(1-self.b2**self.step) / (1-self.b1**self.step)
        step = self.g1 / (np.sqrt(self.g2) + EPS)
        axpy(-step_size, step, self.params)

        # Iterate averaging
        self.p1 *= self.bp1
        axpy(1 - self.bp1, self.params, self.p1)
        return roll2(self.p1, -self.xy) / (1-self.bp1**self.step), loss

    def roll(self, xy):
        """Rolls the optimizer's internal state."""
        if (xy == 0).all():
            return
        self.xy += xy
        self.g1[:] = roll2(self.g1, xy)
        self.g2[:] = roll2(self.g2, xy)
        self.p1[:] = roll2(self.p1, xy)

    def set_params(self, last_iterate):
        """Sets params to the supplied array (a possibly-resized or altered last non-averaged
        iterate), resampling the optimizer's internal state if the shape has changed."""
        self.params = last_iterate
        hw = self.params.shape[-2:]
        self.g1 = resize(self.g1, hw)
        self.g2 = np.maximum(0, resize(self.g2, hw, method=Image.BILINEAR))
        self.p1 = resize(self.p1, hw)

    def restore_state(self, optimizer):
        """Given an AdamOptimizer instance, restores internal state from it."""
        assert isinstance(optimizer, AdamOptimizer)
        self.params = optimizer.params
        self.g1 = optimizer.g1
        self.g2 = optimizer.g2
        self.p1 = optimizer.p1
        self.step = optimizer.step
        self.xy = optimizer.xy.copy()
        self.roll(-self.xy)


FeatureMapRequest = namedtuple('FeatureMapRequest', 'resp img layers')
FeatureMapResponse = namedtuple('FeatureMapResponse', 'resp features')
SCGradRequest = namedtuple('SCGradRequest',
                           '''resp img roll start content_layers style_layers dd_layers
                           layer_weights content_weight style_weight dd_weight''')
SCGradResponse = namedtuple('SCGradResponse', 'resp loss grad')
SetFeaturesAndGrams = namedtuple('SetFeaturesAndGrams', 'features grams')
SetThreadCount = namedtuple('SetThreadCount', 'threads')


class TileWorker:
    """Computes feature maps and gradients on the specified device in a separate process."""
    def __init__(self, req_q, resp_q, model, device=-1):
        self.req_q = req_q
        self.resp_q = resp_q
        self.model = None
        self.model_info = (model.deploy, model.weights, model.mean, model.net_type, model.shapes)
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
            self.process_one_request()

    def process_one_request(self):
        """Receives one request from the master process and acts on it."""
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
                if layer in req.content_layers + req.style_layers + req.dd_layers:
                    layers.append(layer)
            self.model.roll(req.roll, jitter_scale=1)
            loss, grad = self.model.eval_sc_grad_tile(
                req.img.array, req.start, layers, req.content_layers, req.style_layers,
                req.dd_layers, req.layer_weights, req.content_weight, req.style_weight,
                req.dd_weight)
            req.img.unlink()
            self.model.roll(-req.roll, jitter_scale=1)
            self.resp_q.put(SCGradResponse(req.resp, loss, SharedNDArray.copy(grad)))

        if isinstance(req, SetFeaturesAndGrams):
            self.model.features = \
                {layer: req.features[layer].array.copy() for layer in req.features}
            self.model.grams = \
                {layer: req.grams[layer].array.copy() for layer in req.grams}
            self.resp_q.put(())

        if isinstance(req, SetThreadCount):
            set_thread_count(req.threads)


class TileWorkerPoolError(Exception):
    """Indicates abnormal termination of TileWorker processes."""
    pass


class TileWorkerPool:
    """A collection of TileWorkers."""
    def __init__(self, model, devices):
        self.workers = []
        self.req_count = 0
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
        self.req_count += 1
        self.next_worker = (self.next_worker + 1) % len(self.workers)

    def reset_next_worker(self):
        """Sets the worker which will process the next request to worker 0."""
        if MKL_THREADS is not None:
            active_workers = max(1, self.req_count)
            active_workers = min(len(self.workers), active_workers)
            threads_per_process = MKL_THREADS // active_workers
            self.set_thread_count(threads_per_process)
        self.req_count = 0
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

    def set_thread_count(self, threads):
        """Sets the MKL thread count per worker process."""
        for worker in self.workers:
            worker.req_q.put(SetThreadCount(threads))


class CaffeModel:
    """A Caffe neural network model."""
    def __init__(self, deploy, weights, mean=(0, 0, 0), net_type=None, shapes=None,
                 placeholder=False):
        self.deploy = deploy
        self.weights = weights
        self.mean = np.float32(mean).reshape((3, 1, 1))
        self.bgr = True
        self.shapes = shapes
        self.net_type = net_type
        self.last_layer = None
        if shapes:
            self.last_layer = list(shapes)[-1]
        if not placeholder:
            import caffe
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

    def pil_to_image(self, img):
        """Preprocesses a PIL image into params format."""
        arr = np.float32(img).transpose((2, 0, 1))
        if self.bgr:
            arr = arr[::-1]
        return np.ascontiguousarray(arr - self.mean)

    def set_image(self, img):
        """Sets the current model input to a PIL image."""
        self.img = self.pil_to_image(img)

    def resize_image(self, size):
        """Resamples the current model input to a different size."""
        self.img = np.ascontiguousarray(resize(self.img, size[::-1]))

    def layers(self):
        """Returns the layer names of the network."""
        if self.shapes:
            return list(self.shapes)
        layers = []
        for i, layer in enumerate(self.net.blobs.keys()):
            if i == 0:
                continue
            if layer.find('_split_') == -1:
                layers.append(layer)
        return layers

    def layer_info(self, layer):
        """Returns the scale factor vs. the image and the number of channels."""
        return 224 // self.shapes[layer][1], self.shapes[layer][0]

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
            for layer in layers:
                if i == 0:
                    self.features[layer] = feats[layer]
                else:
                    axpy(1, feats[layer], self.features[layer])
            self.roll(-xy)
        for layer in self.features:
            self.features[layer] /= passes
        return self.features

    def preprocess_images(self, pool, content_image, style_images, content_layers, style_layers,
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
        for layer in style_layers:
            _, ch = self.layer_info(layer)
            self.grams[layer] = np.zeros((ch, ch), np.float32)
        for image in style_images:
            self.set_image(image)
            feats = self.prepare_features(pool, style_layers, tile_size)
            for layer in feats:
                axpy(1 / len(style_images), gram_matrix(feats[layer]), self.grams[layer])

        # Prepare feature maps from content image
        print_('Preprocessing the content image...')
        self.set_image(content_image)
        self.prepare_features(pool, content_layers, tile_size)

        return layers

    def eval_sc_grad_tile(self, img, start, layers, content_layers, style_layers, dd_layers,
                          layer_weights, content_weight, style_weight, dd_weight):
        """Evaluates an individual style+content gradient tile."""
        self.net.blobs['data'].reshape(1, 3, *img.shape[-2:])
        self.data['data'] = img
        loss = 0

        # Prepare gradient buffers and run the model forward
        for layer in layers:
            self.diff[layer] = 0
        self.net.forward(end=layers[0])

        for i, layer in enumerate(layers):
            lw = layer_weights[layer]

            # Compute the content and style gradients
            if layer in content_layers:
                scale, _ = self.layer_info(layer)
                start = start // scale
                end = start + np.array(self.data[layer].shape[-2:])
                feat = self.features[layer][:, start[0]:end[0], start[1]:end[1]]
                c_grad = self.data[layer] - feat
                loss += lw * content_weight[layer] * norm2(c_grad)
                axpy(lw * content_weight[layer], normalize(c_grad), self.diff[layer])
            if layer in style_layers:
                current_gram = gram_matrix(self.data[layer])
                n, mh, mw = self.data[layer].shape
                feat = self.data[layer].reshape((n, mh * mw))
                s_grad = blas.ssymm(1, current_gram - self.grams[layer], feat)
                s_grad = s_grad.reshape((n, mh, mw))
                loss += lw * style_weight[layer] * norm2(current_gram - self.grams[layer]) / 2
                axpy(lw * style_weight[layer], normalize(s_grad), self.diff[layer])
            if layer in dd_layers:
                loss -= lw * dd_weight[layer] * norm2(self.data[layer])
                axpy(-lw * dd_weight[layer], normalize(self.data[layer]), self.diff[layer])

            # Run the model backward
            if i+1 == len(layers):
                self.net.backward(start=layer)
            else:
                self.net.backward(start=layer, end=layers[i+1])

        return loss, self.diff['data']

    def eval_sc_grad(self, pool, roll, content_layers, style_layers, dd_layers, layer_weights,
                     content_weight, style_weight, dd_weight, tile_size):
        """Evaluates the summed style and content gradients."""
        loss = 0
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
                                  content_layers, style_layers, dd_layers, layer_weights,
                                  content_weight, style_weight, dd_weight))
        pool.reset_next_worker()
        for _ in range(np.prod(ntiles)):
            (start, end), loss_tile, grad_tile = pool.resp_q.get()
            loss += loss_tile
            grad[:, start[0]:end[0], start[1]:end[1]] = grad_tile.array
            grad_tile.unlink()

        return loss, grad

    def roll(self, xy, jitter_scale=32):
        """Rolls image and feature maps."""
        xy = xy * jitter_scale
        if (xy == 0).all():
            return
        for layer, feat in self.features.items():
            scale, _ = self.layer_info(layer)
            self.features[layer][:] = roll2(feat, xy // scale)
        self.img[:] = roll2(self.img, xy)


class StyleTransfer:
    """Performs style transfer."""
    def __init__(self, model):
        self.model = model
        self.layer_weights = {layer: 1.0 for layer in self.model.layers() + ['data']}
        if ARGS.layer_weights:
            with open(ARGS.layer_weights) as lw_file:
                self.layer_weights.update(json.load(lw_file))
        self.aux_image = None
        self.current_output = None
        self.current_raw = None
        self.optimizer = None
        self.pool = None
        self.step = 0

    @staticmethod
    def parse_weights(args, master_weight):
        """Parses a list of name:number pairs into a normalized dict of weights."""
        names = []
        weights = {}
        total = 0
        for arg in args:
            name, _, w = arg.partition(':')
            names.append(name)
            if w:
                weights[name] = ffloat(w)
            else:
                weights[name] = 1
            total += abs(weights[name])
        return names, {name: weight * master_weight / total for name, weight in weights.items()}

    def eval_loss_and_grad(self, img, sc_grad_args):
        """Returns the summed loss and gradient."""
        old_img = self.model.img
        self.model.img = img
        lw = self.layer_weights['data']

        # Compute style+content gradient
        loss, grad = self.model.eval_sc_grad(*sc_grad_args)
        normalize(grad)

        # Compute total variation gradient
        tv_loss, tv_grad = tv_norm(self.model.img / 255, beta=ARGS.tv_power)
        loss += lw * ARGS.tv_weight * tv_loss

        # Selectively blur edges more to obscure jitter and tile seams
        tv_mask = np.ones_like(tv_grad)
        tv_mask[:, :2, :] = 5
        tv_mask[:, -2:, :] = 5
        tv_mask[:, :, :2] = 5
        tv_mask[:, :, -2:] = 5
        tv_grad *= tv_mask
        axpy(lw * ARGS.tv_weight, tv_grad, grad)

        # Compute p-norm regularizer gradient (from jcjohnson/cnn-vis and [3])
        p = ARGS.p_power
        img_scaled = abs(self.model.img / 127.5)
        img_pow = img_scaled**(p-1)
        loss += lw * ARGS.p_weight * np.sum(img_pow * img_scaled)
        p_grad = p * np.sign(self.model.img) * img_pow
        axpy(lw * ARGS.p_weight, p_grad, grad)

        # Compute auxiliary image gradient
        if self.aux_image is not None:
            aux_grad = (self.model.img - self.aux_image) / 255
            loss += lw * ARGS.aux_weight * norm2(aux_grad)
            axpy(lw * ARGS.aux_weight, aux_grad, grad)

        self.model.img = old_img
        return loss, grad

    def transfer(self, iterations, params, content_image, style_images, callback=None):
        """Performs style transfer from style_image to content_image."""
        content_layers, content_weight = self.parse_weights(ARGS.content_layers,
                                                            ARGS.content_weight)
        style_layers, style_weight = self.parse_weights(ARGS.style_layers, 1)
        dd_layers, dd_weight = self.parse_weights(ARGS.dd_layers, ARGS.dd_weight)

        layers = self.model.preprocess_images(
            self.pool, content_image, style_images, content_layers, style_layers, ARGS.tile_size)
        self.pool.set_features_and_grams(self.model.features, self.model.grams)
        self.model.img = params

        old_img = self.model.img.copy()
        self.step += 1
        log = open('log.csv', 'w')
        print_('step', 'loss', 'img_size', 'update_size', 'tv_loss', sep=',', file=log, flush=True)

        for step in range(1, iterations+1):
            # Forward jitter
            jitter_scale, _ = self.model.layer_info([l for l in layers if l in content_layers][0])
            xy = np.array((0, 0))
            img_size = np.array(self.model.img.shape[-2:])
            if max(*img_size) > ARGS.tile_size:
                xy = np.int32(np.random.uniform(-0.5, 0.5, size=2) * img_size) // jitter_scale
            self.model.roll(xy, jitter_scale=jitter_scale)
            self.optimizer.roll(xy * jitter_scale)

            # In-place gradient descent update
            args = (self.pool, xy * jitter_scale, content_layers, style_layers, dd_layers,
                    self.layer_weights, content_weight, style_weight, dd_weight, ARGS.tile_size)
            avg_img, loss = self.optimizer.update(partial(self.eval_loss_and_grad,
                                                          sc_grad_args=args))

            # Backward jitter
            self.model.roll(-xy, jitter_scale=jitter_scale)
            self.optimizer.roll(-xy * jitter_scale)

            # Compute image size statistic
            img_size = np.mean(abs(avg_img))

            # Compute update size statistic
            update_size = np.mean(abs(avg_img - old_img))
            old_img[:] = avg_img

            # Compute total variation statistic
            x_diff = avg_img - np.roll(avg_img, -1, axis=-1)
            y_diff = avg_img - np.roll(avg_img, -1, axis=-2)
            tv_loss = np.sum(x_diff**2 + y_diff**2) / avg_img.size

            # Record current output
            self.current_raw = avg_img
            self.current_output = self.model.get_image(avg_img)

            print_(step, loss / avg_img.size, img_size, update_size, tv_loss, sep=',', file=log,
                   flush=True)

            if callback is not None:
                callback(step=step, update_size=update_size, loss=loss / avg_img.size,
                         tv_loss=tv_loss)

        return self.current_output

    def transfer_multiscale(self, sizes, iterations, content_image, style_images, initial_image,
                            aux_image, initial_state=None, **kwargs):
        """Performs style transfer from style_image to content_image at the given sizes."""
        output_image = None
        output_raw = None
        print_('Starting %d worker process(es).' % len(ARGS.devices))
        self.pool = TileWorkerPool(self.model, ARGS.devices)

        for i, size in enumerate(sizes):
            content_scaled = resize_to_fit(content_image, size, scale_up=True)
            style_scaled = []
            for image in style_images:
                style_scaled.append(resize_to_fit(image, round(size * ARGS.style_scale),
                                                  scale_up=ARGS.style_scale_up))
            if aux_image:
                aux_scaled = aux_image.resize(content_scaled.size, Image.LANCZOS)
                self.aux_image = self.model.pil_to_image(aux_scaled)
            if output_image:  # this is not the first scale
                self.model.img = output_raw
                self.model.resize_image(content_scaled.size)
                params = self.model.img
                self.optimizer.set_params(params)
            else:  # this is the first scale
                if initial_image:  # and the user supplied an initial image
                    initial_image = initial_image.resize(content_scaled.size, Image.LANCZOS)
                    self.model.set_image(initial_image)
                else:  # and the user did not supply an initial image
                    w, h = content_scaled.size
                    self.model.set_image(np.random.uniform(0, 255, size=(h, w, 3)))

                # make sure the optimizer's params array shares memory with self.model.img
                # after preprocess_image is called later
                self.optimizer = AdamOptimizer(
                    self.model.img, step_size=ARGS.step_size, bp1=1-(1/ARGS.avg_window))

                if initial_state:
                    self.optimizer.restore_state(initial_state)
                    if self.model.img.shape != self.optimizer.params.shape:
                        initial_image = self.model.get_image(self.optimizer.params)
                        initial_image = initial_image.resize(content_scaled.size, Image.LANCZOS)
                        self.model.set_image(initial_image)
                        self.optimizer.set_params(self.model.img)
                    self.model.img = self.optimizer.params

            params = self.model.img
            iters_i = iterations[min(i, len(iterations)-1)]
            output_image = self.transfer(iters_i, params, content_scaled, style_scaled, **kwargs)
            output_raw = self.current_raw

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
    tv_loss = np.nan

    def __init__(self, transfer, url=None, steps=-1, save_every=0):
        self.transfer = transfer
        self.url = url
        self.steps = steps
        self.save_every = save_every

    def __call__(self, step=-1, update_size=np.nan, loss=np.nan, tv_loss=np.nan):
        this_t = timer()
        self.step += 1
        self.update_size = update_size
        self.loss = loss
        self.tv_loss = tv_loss
        if self.save_every and self.step % self.save_every == 0:
            self.transfer.current_output.save('out_%04d.png' % self.step)
        if self.step == 1:
            if self.url:
                webbrowser.open(self.url)
        else:
            self.t = this_t - self.prev_t
        print_('Step %d, time: %.2f s, update: %.2f, loss: %.1f, tv: %.1f' %
               (step, self.t, update_size, loss, tv_loss), flush=True)
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
    <p>Step %(step)d/%(steps)d, time: %(t).2f s/step, update: %(update_size).2f, loss: %(loss).1f,
    tv: %(tv_loss).1f
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
                'tv_loss': self.server.progress.tv_loss,
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
    """Parses command line arguments. Alternate default arguments are read from style_transfer.ini
    (an alternate config can be specified by --config). The .ini file should begin with the
    line [DEFAULT] and contain keys corresponding to the long option names."""
    config_file = 'style_transfer.ini'

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('content_image', help='the content image')
    parser.add_argument('style_image', help='the style image')
    parser.add_argument('output_image', nargs='?', default='out.png', help='the output image')
    parser.add_argument('--config', default=config_file,
                        help='an ini file containing values for command line arguments')
    parser.add_argument('--init-image', metavar='IMAGE', help='the initial image')
    parser.add_argument('--aux-image', metavar='IMAGE', help='the auxiliary image')
    parser.add_argument('--state', help='a .state file (the initial state)')
    parser.add_argument(
        '--iterations', '-i', nargs='+', type=int, default=[300], help='the number of iterations')
    parser.add_argument(
        '--size', '-s', nargs='+', type=int, default=[256], help='the output size(s)')
    parser.add_argument(
        '--style-scale', '-ss', type=ffloat, default=1, help='the style scale factor')
    parser.add_argument(
        '--style-scale-up', default=False, action='store_true',
        help='allow scaling style image up')
    parser.add_argument(
        '--step-size', '-st', type=ffloat, default=15,
        help='the Adam step size (iteration magnitude)')
    parser.add_argument(
        '--avg-window', type=ffloat, default=20, help='the iterate averaging window size')
    parser.add_argument(
        '--layer-weights', help='a json file containing per-layer weight scaling factors')
    parser.add_argument(
        '--content-weight', '-cw', type=ffloat, default=0.05, help='the content image factor')
    parser.add_argument(
        '--dd-weight', '-dw', type=ffloat, default=0, help='the Deep Dream factor')
    parser.add_argument(
        '--tv-weight', '-tw', type=ffloat, default=1, help='the smoothing factor')
    parser.add_argument(
        '--tv-power', '-tp', metavar='BETA', type=ffloat, default=2, help='the smoothing exponent')
    parser.add_argument(
        '--p-weight', '-pw', type=ffloat, default=0.05, help='the p-norm regularizer factor')
    parser.add_argument(
        '--p-power', '-pp', metavar='P', type=ffloat, default=6, help='the p-norm exponent')
    parser.add_argument(
        '--aux-weight', '-aw', type=ffloat, default=1, help='the auxiliary image factor')
    parser.add_argument(
        '--content-layers', nargs='*', default=['conv4_2'],
        metavar='LAYER', help='the layers to use for content')
    parser.add_argument(
        '--style-layers', nargs='*', metavar='LAYER',
        default=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        help='the layers to use for style')
    parser.add_argument(
        '--dd-layers', nargs='*', metavar='LAYER', default=[],
        help='the layers to use for Deep Dream')
    parser.add_argument(
        '--port', '-p', type=int, default=8000,
        help='the port to use for the http server')
    parser.add_argument(
        '--no-browser', action='store_true', help='don\'t open a web browser')
    parser.add_argument(
        '--hidpi', action='store_true', help='display the image at 2x scale in the browser')
    parser.add_argument(
        '--model', default='vgg19.prototxt',
        help='the Caffe deploy.prototxt for the model to use')
    parser.add_argument(
        '--weights', default='vgg19.caffemodel',
        help='the Caffe .caffemodel for the model to use')
    parser.add_argument(
        '--mean', nargs=3, metavar=('B_MEAN', 'G_MEAN', 'R_MEAN'),
        default=(103.939, 116.779, 123.68),
        help='the per-channel means of the model (BGR order)')
    parser.add_argument(
        '--save-every', metavar='N', type=int, default=0, help='save the image every n steps')
    parser.add_argument(
        '--devices', nargs='+', metavar='DEVICE', type=int, default=[0],
        help='device numbers to use (-1 for cpu)')
    parser.add_argument(
        '--tile-size', type=int, default=512, help='the maximum rendering tile size')
    parser.add_argument(
        '--seed', type=int, default=0, help='the random seed')
    parser.add_argument(
        '--list-layers', action='store_true', help='list the model\'s layers')

    global ARGS  # pylint: disable=global-statement
    args_from_cli = parser.parse_args()
    config = configparser.ConfigParser()
    if os.path.exists(args_from_cli.config) or args_from_cli.config != config_file:
        config.read_file(open(args_from_cli.config))
    config_args = [None, None]
    for k, v in config['DEFAULT'].items():
        config_args.append('--' + k.replace('_', '-'))
        if v:
            config_args.extend(shlex.split(v))
    config_parsed = parser.parse_args(args=config_args)
    new_defaults = {arg: getattr(config_parsed, arg) for arg in config['DEFAULT']}
    ARGS = parser.parse_args(namespace=argparse.Namespace(**new_defaults))


def print_args():
    """Prints out all command-line parameters."""
    print_('Parameters:')
    for item in sorted(vars(ARGS).items()):
        print_('% 14s: %s' % item)
    print_()


def get_image_comment():
    """Makes a comment string to write into the output image."""
    s = 'Created with https://github.com/crowsonkb/style_transfer.\n\n'
    s += 'Command line: style_transfer.py ' + ' '.join(sys.argv[1:]) + '\n\n'
    s += 'Parameters:\n'
    for item in sorted(vars(ARGS).items()):
        s += '%s: %s\n' % item
    return s


def init_model(resp_q, net_type):
    """Puts the list of layer shapes into resp_q. To be run in a separate process."""
    import caffe
    caffe.set_mode_cpu()
    model = CaffeModel(ARGS.model, ARGS.weights, ARGS.mean, net_type)
    shapes = OrderedDict()
    for layer in model.layers():
        shapes[layer] = model.data[layer].shape
    resp_q.put(shapes)


def main():
    """CLI interface for style transfer."""
    start_time = timer()
    parse_args()
    print_args()

    if MKL_THREADS is not None:
        print_('MKL detected, %d threads maximum.\n' % MKL_THREADS)

    os.environ['GLOG_minloglevel'] = '2'

    print_('Loading %s.' % ARGS.weights)
    resp_q = CTX.Queue()
    CTX.Process(target=init_model, args=(resp_q, None)).start()
    shapes = resp_q.get()
    model = CaffeModel(ARGS.model, ARGS.weights, ARGS.mean, None, shapes=shapes,
                       placeholder=True)
    transfer = StyleTransfer(model)
    if ARGS.list_layers:
        print_('Layers:')
        for layer, shape in model.shapes.items():
            print_('% 25s %s' % (layer, shape))
        sys.exit(0)

    sizes = sorted(ARGS.size)
    content_image = Image.open(ARGS.content_image).convert('RGB')
    style_images = []
    for image in ARGS.style_image.split(','):
        style_images.append(Image.open(image).convert('RGB'))
    initial_image, aux_image = None, None
    if ARGS.init_image:
        initial_image = Image.open(ARGS.init_image).convert('RGB')
    if ARGS.aux_image:
        aux_image = Image.open(ARGS.aux_image).convert('RGB')

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
            sizes, ARGS.iterations, content_image, style_images, initial_image, aux_image,
            callback=server.progress, initial_state=state)
    except KeyboardInterrupt:
        print_()

    if transfer.current_output:
        print_('Saving output as %s.' % ARGS.output_image)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_itxt('Comment', get_image_comment())
        transfer.current_output.save(ARGS.output_image, pnginfo=png_info)
        a, _, _ = ARGS.output_image.rpartition('.')
        print_('Saving state as %s.' % (a + '.state'))
        transfer.save_state(a + '.state')
    time_spent = timer() - start_time
    print_('Exiting after %dm %.2fs.' % (time_spent // 60, time_spent % 60))

if __name__ == '__main__':
    main()
