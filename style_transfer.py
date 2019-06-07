#!/usr/bin/env python3

"""Neural style transfer using Caffe. Implements A Neural Algorithm of Artistic Style
(http://arxiv.org/abs/1508.06576)."""

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, too-many-locals
# pylint: disable=global-statement

from __future__ import division

from argparse import Namespace
from collections import namedtuple
import csv
from datetime import datetime
from functools import partial
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import webbrowser

import numpy as np
from PIL import Image, PngImagePlugin
from shared_ndarray import SharedNDArray

from config_system import ffloat, parse_args
import log_utils
from num_utils import (saxpy, gram_matrix, norm2, normalize, p_norm, resize,
                       roll2, ssymm, tv_norm, swt_norm, consume)
from optimizers import AdamOptimizer, LBFGSOptimizer
import web_interface


ARGS = None

# Use forking mode for multiprocessing
CTX = mp.get_context('fork')

# Maximum number of MKL threads between all processes
MKL_THREADS = None

# Run identifier
RUN = ''

# State object for configuration file
STATE = Namespace()


def terminal_bg(light=True, dark=False, default=None):
    """Returns the first argument if the terminal has a light background, the second if it has a
    dark background, and the third if it cannot determine."""
    colorfgbg = os.environ.get('COLORFGBG', '')
    if colorfgbg == '0;15':
        return light
    if colorfgbg == '15;0':
        return dark
    return default


def setup_exceptions():
    scheme = terminal_bg('LightBG', 'Linux', 'Neutral')
    mode = 'Plain'
    if 'DEBUG' in os.environ:
        mode = 'Verbose'
    try:
        from IPython.core.ultratb import AutoFormattedTB
        sys.excepthook = AutoFormattedTB(mode=mode, color_scheme=scheme)
    except ImportError:
        pass


logger = log_utils.setup_logger('style_transfer')


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


class StatLogger:
    """Collects per-iteration statistics to be written to a CSV file on exit."""
    def __init__(self):
        self.lock = CTX.Lock()
        self.stats = []
        self.start_time = None

    def update_current_it(self, **kwargs):
        with self.lock:
            self.stats[-1].update(kwargs)

    def update_new_it(self, **kwargs):
        with self.lock:
            if self.start_time is None:
                self.start_time = time.perf_counter()
            self.stats.append(kwargs)
            self.stats[-1]['iteration'] = len(self.stats) - 1
            self.stats[-1]['time'] = time.perf_counter() - self.start_time

    def dump(self):
        with self.lock, open(RUN + '_log.csv', 'w', newline='') as f:
            fields = ['iteration', 'scale', 'step', 'time']
            for row in self.stats:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.stats)


STATS = None


class LayerIndexer:
    """Helper class for accessing feature maps and gradients."""
    def __init__(self, net, attr):
        self.net, self.attr = net, attr

    def __getitem__(self, key):
        return getattr(self.net.blobs[key], self.attr)[0]

    def __setitem__(self, key, value):
        getattr(self.net.blobs[key], self.attr)[0] = value


FeatureMapRequest = namedtuple('FeatureMapRequest', 'resp img layers')
FeatureMapResponse = namedtuple('FeatureMapResponse', 'resp features')
SCGradRequest = namedtuple('SCGradRequest',
                           '''resp img roll start content_layers style_layers dd_layers
                           layer_weights content_weight style_weight dd_weight''')
SCGradResponse = namedtuple('SCGradResponse', 'resp loss grad')
SetContentsAndStyles = namedtuple('SetContentsAndStyles', 'contents styles')
SetThreadCount = namedtuple('SetThreadCount', 'threads')

ContentData = namedtuple('ContentData', 'features')
StyleData = namedtuple('StyleData', 'grams')


class TileWorker:
    """Computes feature maps and gradients on the specified device in a separate process."""
    def __init__(self, req_q, resp_q, model, device=-1, caffe_path=None):
        self.req_q = req_q
        self.resp_q = resp_q
        self.model = None
        self.model_info = (model.deploy, model.weights, model.mean, model.shapes)
        self.device = device
        self.caffe_path = caffe_path
        self.proc = CTX.Process(target=self.run)
        self.proc.daemon = True
        self.proc.start()

    def __del__(self):
        if not self.proc.exitcode:
            self.proc.terminate()

    def run(self):
        """This method runs in the new process."""
        global logger
        setup_exceptions()
        logger = log_utils.setup_logger('tile_worker')

        if self.caffe_path is not None:
            sys.path.append(self.caffe_path + '/python')
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
            try:
                self.process_one_request()
            except KeyboardInterrupt:
                break

    def process_one_request(self):
        """Receives one request from the master process and acts on it."""
        req = self.req_q.get()
        logger.debug('Started request %s', req)
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

        if isinstance(req, SetContentsAndStyles):
            self.model.contents, self.model.styles = [], []

            for content in req.contents:
                features = \
                    {layer: content.features[layer].array.copy() for layer in content.features}
                self.model.contents.append(ContentData(features))
            for style in req.styles:
                grams = \
                    {layer: style.grams[layer].array.copy() for layer in style.grams}
                self.model.styles.append(StyleData(grams))
            self.resp_q.put(())

        if isinstance(req, SetThreadCount):
            set_thread_count(req.threads)

        logger.debug('Finished request %s', req)


class TileWorkerPoolError(Exception):
    """Indicates abnormal termination of TileWorker processes."""
    pass


class TileWorkerPool:
    """A collection of TileWorkers."""
    def __init__(self, model, devices, caffe_path=None):
        self.workers = []
        self.req_count = 0
        self.next_worker = 0
        self.resp_q = CTX.Queue()
        self.is_healthy = True
        for device in devices:
            self.workers.append(TileWorker(CTX.Queue(), self.resp_q, model, device, caffe_path))

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

    def set_contents_and_styles(self, contents, styles):
        """Propagates feature maps and Gram matrices to all TileWorkers."""
        content_shms, style_shms = [], []

        for content in contents:
            features_shm = {layer: SharedNDArray.copy(content.features[layer])
                            for layer in content.features}
            content_shms.append(ContentData(features_shm))

        for style in styles:
            grams_shm = {layer: SharedNDArray.copy(style.grams[layer])
                         for layer in style.grams}
            style_shms.append(StyleData(grams_shm))

        for worker in self.workers:
            worker.req_q.put(SetContentsAndStyles(content_shms, style_shms))

        for worker in self.workers:
            self.resp_q.get()

        for shms in content_shms:
            consume(shm.unlink() for shm in shms.features.values())
        for shms in style_shms:
            consume(shm.unlink() for shm in shms.grams.values())

    def set_thread_count(self, threads):
        """Sets the MKL thread count per worker process."""
        for worker in self.workers:
            worker.req_q.put(SetThreadCount(threads))


class ArrayPool:
    """A pool of preallocated (C-contiguous) NumPy arrays."""
    def __init__(self):
        self.pool = {}

    def array(self, shape, dtype):
        key = (shape, dtype)
        if key not in self.pool:
            self.pool[key] = np.zeros(shape, dtype)
        return self.pool[key]

    def array_like(self, arr):
        return self.array(arr.shape, arr.dtype)


class CaffeModel:
    """A Caffe neural network model."""
    def __init__(self, deploy, weights, mean=(0, 0, 0), shapes=None, placeholder=False):
        self.deploy = deploy
        self.weights = weights
        self.mean = np.float32(mean).reshape((3, 1, 1))
        self.bgr = True
        self.shapes = shapes
        self.last_layer = None
        if shapes:
            self.last_layer = list(shapes)[-1]
        if not placeholder:
            import caffe
            self.net = caffe.Net(self.deploy, 1, weights=self.weights)
            self.data = LayerIndexer(self.net, 'data')
            self.diff = LayerIndexer(self.net, 'diff')
        self.contents = []
        self.styles = []
        self.img = None
        self._arr_pool = ArrayPool()

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
        if len(self.shapes[layer]) == 1:
            return 224, self.shapes[layer][0]
        return 224 // self.shapes[layer][1], self.shapes[layer][0]

    def eval_features_tile(self, img, layers):
        """Computes a single tile in a set of feature maps."""
        self.net.blobs['data'].reshape(1, 3, *img.shape[-2:])
        self.data['data'] = img
        self.net.forward(end=self.last_layer)
        self.data[self.last_layer] = np.maximum(0, self.data[self.last_layer])
        return {layer: self.data[layer] for layer in layers}

    def eval_features_once(self, pool, layers, tile_size=512):
        """Computes the set of feature maps for an image."""
        img_size = np.array(self.img.shape[-2:])
        ntiles = (img_size - 1) // tile_size + 1
        tile_size = img_size // ntiles
        if np.prod(ntiles) > 1:
            print('Using %dx%d tiles of size %dx%d.' %
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
        features = {}
        for i in range(passes):
            xy = np.array((0, 0))
            if i > 0:
                xy = np.int32(np.random.uniform(size=2) * img_size) // 32
            self.roll(xy)
            self.roll_features(features, xy)
            feats = self.eval_features_once(pool, layers, tile_size)
            for layer in layers:
                if i == 0:
                    features[layer] = feats[layer] / passes
                else:
                    saxpy(1 / passes, feats[layer], features[layer])
            self.roll(-xy)
            self.roll_features(features, -xy)
        return features

    def preprocess_images(self, pool, content_images, style_images, content_layers, style_layers,
                          tile_size=512, roll=None):
        """Performs preprocessing tasks on the input images."""
        # Construct list of layers to visit during the backward pass
        layers = []
        for layer in reversed(self.layers()):
            if layer in content_layers or layer in style_layers:
                layers.append(layer)

        # Prepare Gram matrices from style images
        if roll is None:
            print('Preprocessing the style image(s)...')

        sizes = [None]
        if ARGS.style_multiscale:
            vmin, vmax = ARGS.style_multiscale
            size = vmax
            sizes = [vmax]
            while True:
                size = int(round(size / np.sqrt(2)))
                if size < max(32, vmin):
                    break
                sizes.append(size)

        if not self.styles:
            grams = {}
            count = 0
            for i, image in enumerate(style_images):
                scale_too_big_cond = False
                for size in reversed(sizes):
                    if scale_too_big_cond:
                        break
                    if size:
                        image_scaled = resize_to_fit(image, size)
                        if max(image_scaled.size) == max(image.size):
                            scale_too_big_cond = True
                        if min(image_scaled.size) < 32:
                            continue
                        self.set_image(image_scaled)
                        h, w = image_scaled.size
                        print('Processing style {} at {}x{}.'.format(i + 1, h, w))
                    else:
                        self.set_image(image)
                    roll2(self.img, roll)
                    feats = self.prepare_features(pool, style_layers, tile_size, passes=1)
                    for layer in feats:
                        gram = gram_matrix(feats[layer])
                        if layer not in grams:
                            grams[layer] = gram
                        else:
                            grams[layer] += gram
                    count += 1
            for gram in grams.values():
                gram /= count
            self.styles.append(StyleData(grams))

        # Prepare feature maps from content image
        if roll is None:
            print('Preprocessing the content image(s)...')
            n = 10
        else:
            n = 1
        for image in content_images:
            self.set_image(image)
            roll2(self.img, roll)
            feats = self.prepare_features(pool, content_layers, tile_size, passes=n)
            self.contents.append(ContentData(feats))

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
        self.data[layers[0]] = np.maximum(0, self.data[layers[0]])

        for i, layer in enumerate(layers):
            lw = layer_weights[layer]
            scale, _ = self.layer_info(layer)
            start_ = start // scale
            end = start_ + np.array(self.data[layer].shape[-2:])

            def eval_c_grad(layer, content):
                nonlocal loss
                feat = content.features[layer][:, start_[0]:end[0], start_[1]:end[1]]
                c_grad = self.data[layer] - feat
                loss += lw * content_weight[layer] * norm2(c_grad)
                saxpy(lw * content_weight[layer], normalize(c_grad), self.diff[layer])

            def eval_s_grad(layer, style):
                nonlocal loss
                current_gram = gram_matrix(self.data[layer])
                n, mh, mw = self.data[layer].shape
                feat = self.data[layer].reshape((n, mh * mw))
                gram_diff = current_gram - style.grams[layer]
                s_grad = self._arr_pool.array_like(feat)
                ssymm(gram_diff, feat, c=s_grad)
                s_grad = s_grad.reshape((n, mh, mw))
                loss += lw * style_weight[layer] * norm2(gram_diff) / len(self.styles)
                saxpy(lw * style_weight[layer] / len(self.styles), normalize(s_grad),
                     self.diff[layer])

            # Compute the content and style gradients
            if layer in content_layers:
                for content in self.contents:
                    eval_c_grad(layer, content)
            if layer in style_layers:
                for style in self.styles:
                    eval_s_grad(layer, style)
            if layer in dd_layers:
                loss -= lw * dd_weight[layer] * norm2(self.data[layer])
                saxpy(-lw * dd_weight[layer], normalize(self.data[layer]), self.diff[layer])

            # Run the model backward
            if i + 1 == len(layers):
                self.net.backward(start=layer)
            else:
                self.net.backward(start=layer, end=layers[i + 1])

        return loss, self.diff['data']

    def eval_sc_grad(self, pool, roll, content_layers, style_layers, dd_layers, layer_weights,
                     content_weight, style_weight, dd_weight, tile_size):
        """Evaluates the summed style and content gradients."""
        loss = 0
        grad = np.zeros_like(self.img)
        img_size = np.array(self.img.shape[-2:])
        ntiles = (img_size - 1) // tile_size + 1
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

    def roll_features(self, feats, xy, jitter_scale=32):
        """Rolls an individual set of feature maps in-place."""
        xy = xy * jitter_scale

        for layer, feat in feats.items():
            scale, _ = self.layer_info(layer)
            roll2(feat, xy // scale)

        return feats

    def roll(self, xy, jitter_scale=32):
        """Rolls the image, feature maps."""
        for content in self.contents:
            self.roll_features(content.features, xy, jitter_scale)
        roll2(self.img, xy * jitter_scale)


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
        self.window = None
        if ARGS.display == 'gui':
            from display_image import ImageWindow
            self.window = ImageWindow()

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

        # Compute total variation gradient
        if ARGS.tv_weight:
            tv_loss, tv_grad = tv_norm(self.model.img / 127.5, beta=ARGS.tv_power)
            loss += lw * ARGS.tv_weight * tv_loss
            saxpy(lw * ARGS.tv_weight, tv_grad, grad)

        # Compute SWT norm and gradient
        if ARGS.swt_weight:
            swt_loss, swt_grad = swt_norm(self.model.img / 127.5,
                                          ARGS.swt_wavelet, ARGS.swt_levels, p=ARGS.swt_power)
            loss += lw * ARGS.swt_weight * swt_loss
            saxpy(lw * ARGS.swt_weight, swt_grad, grad)

        # Compute p-norm regularizer gradient (from jcjohnson/cnn-vis and [3])
        if ARGS.p_weight:
            p_loss, p_grad = p_norm((self.model.img + self.model.mean - 127.5) / 127.5,
                                    p=ARGS.p_power)
            loss += lw * ARGS.p_weight * p_loss
            saxpy(lw * ARGS.p_weight, p_grad, grad)

        # Compute auxiliary image gradient
        if self.aux_image is not None:
            aux_grad = (self.model.img - self.aux_image) / 127.5
            loss += lw * ARGS.aux_weight * norm2(aux_grad)
            saxpy(lw * ARGS.aux_weight, aux_grad, grad)

        self.model.img = old_img
        return loss, grad

    def transfer(self, iterations, params, content_images, style_images, callback=None):
        """Performs style transfer from style_image to content_image."""
        if 'scale' not in STATE:
            STATE.scale = 0
        else:
            STATE.scale += 1
        STATE.step = 0
        STATE.steps = iterations
        STATE.img_size = self.model.img.shape[1:]

        content_layers, content_weight = self.parse_weights(ARGS.content_layers,
                                                            ARGS.content_weight)
        style_layers, style_weight = self.parse_weights(ARGS.style_layers, 1)
        dd_layers, dd_weight = self.parse_weights(ARGS.dd_layers, ARGS.dd_weight)

        self.model.contents = []
        if not ARGS.style_multiscale:
            self.model.styles = []

        if ARGS.jitter:
            self.model.preprocess_images(self.pool, [], style_images, [], style_layers,
                                         ARGS.tile_size)

        else:
            self.model.preprocess_images(
                self.pool, content_images, style_images, content_layers, style_layers,
                ARGS.tile_size)
        self.pool.set_contents_and_styles(self.model.contents, self.model.styles)
        self.model.img = params

        old_img = self.model.img.copy()
        self.step += 1

        for step in range(1, iterations + 1):
            STATE.step = step - 1
            STATS.update_new_it(scale=STATE.scale, step=step - 1,
                                content_h=self.model.img.shape[1],
                                content_w=self.model.img.shape[2])

            # Forward jitter
            jitter_scale, _ = self.model.layer_info(
                [l for l in reversed(self.model.layers()) if l in content_layers][0])
            if ARGS.jitter:
                jitter_scale = 1
                self.model.contents = []
            img_size = np.array(self.model.img.shape[-2:])
            xy = np.int32(np.random.uniform(-0.5, 0.5, size=2) * img_size) // jitter_scale
            self.model.roll(xy, jitter_scale=jitter_scale)
            self.optimizer.roll(xy * jitter_scale)

            xy_ = xy
            if ARGS.jitter:
                self.model.preprocess_images(self.pool, content_images, [], content_layers, [],
                                             ARGS.tile_size, roll=xy)
                self.model.img = params
                self.pool.set_contents_and_styles(self.model.contents, self.model.styles)
                xy_ = np.asarray((0, 0))

            # In-place gradient descent update
            args = (self.pool, xy_ * jitter_scale, content_layers, style_layers, dd_layers,
                    self.layer_weights, content_weight, style_weight, dd_weight, ARGS.tile_size)
            avg_img, loss = self.optimizer.update(partial(self.eval_loss_and_grad,
                                                          sc_grad_args=args))

            # Backward jitter
            if ARGS.jitter:
                self.model.contents = []
            self.model.roll(-xy, jitter_scale=jitter_scale)
            self.optimizer.roll(-xy * jitter_scale)

            # Compute update size statistic
            update_size = np.mean(abs(avg_img - old_img))
            old_img[...] = avg_img

            # Compute total variation statistic
            x_diff = avg_img - np.roll(avg_img, -1, axis=-1)
            y_diff = avg_img - np.roll(avg_img, -1, axis=-2)
            tv_loss = np.sqrt(np.mean(x_diff**2 + y_diff**2))

            STATS.update_current_it(update_size=update_size, loss=loss, tv_norm=tv_loss)

            # Record current output
            self.current_raw = avg_img
            self.current_output = self.model.get_image(avg_img)

            if callback is not None:
                msg = callback(step=step, update_size=update_size, loss=loss, tv_loss=tv_loss,
                               image=self.current_output)

            if self.window is not None:
                self.window.display(self.current_output)

        return self.current_output

    def transfer_multiscale(self, content_images, style_images, initial_image, aux_image,
                            callback=None, **kwargs):
        """Performs style transfer from style_image to content_image at the given sizes."""
        output_image = None
        output_raw = None
        print('Starting %d worker process(es).' % len(ARGS.devices))
        self.pool = TileWorkerPool(self.model, ARGS.devices, ARGS.caffe_path)

        size = ARGS.size
        sizes = [ARGS.size]
        while True:
            size = round(size / np.sqrt(2))
            if size < ARGS.min_size:
                break
            sizes.append(size)

        steps = 0
        for i in range(len(sizes)):
            steps += ARGS.iterations[min(i, len(ARGS.iterations) - 1)]
        if callback is not None:
            callback.set_steps(steps)

        for i, size in enumerate(reversed(sizes)):
            content_scaled = []
            for image in content_images:
                if image.size != content_images[0].size:
                    raise ValueError('All of the content images must be the same size')
                content_scaled.append(resize_to_fit(image, size, scale_up=True))
                w, h = content_scaled[0].size
            print('\nScale %d, image size %dx%d.\n' % (i + 1, w, h))
            style_scaled = []
            for image in style_images:
                if ARGS.style_multiscale:
                    style_scaled.append(image)
                elif ARGS.style_scale >= 32:
                    style_scaled.append(resize_to_fit(image, ARGS.style_scale, scale_up=True))
                else:
                    style_size = round(size * ARGS.style_scale)
                    if ARGS.max_style_size is not None:
                        style_size = min(style_size, ARGS.max_style_size)
                    style_scaled.append(resize_to_fit(image, style_size,
                                                      scale_up=ARGS.style_scale_up))
            if aux_image:
                aux_scaled = aux_image.resize(content_scaled[0].size, Image.LANCZOS)
                self.aux_image = self.model.pil_to_image(aux_scaled)
            if output_image:  # this is not the first scale
                self.model.img = output_raw
                self.model.resize_image(content_scaled[0].size)
                params = self.model.img
                self.optimizer.set_params(params)
            else:  # this is the first scale
                biased_g1 = True
                if initial_image:  # and the user supplied an initial image
                    initial_image = initial_image.resize(content_scaled[0].size, Image.LANCZOS)
                    self.model.set_image(initial_image)
                else:  # and the user did not supply an initial image
                    w, h = content_scaled[0].size
                    self.model.set_image(np.random.uniform(0, 255, size=(h, w, 3)))
                    biased_g1 = False
                # make sure the optimizer's params array shares memory with self.model.img
                # after preprocess_image is called later
                if ARGS.optimizer == 'adam':
                    self.optimizer = AdamOptimizer(
                        self.model.img, step_size=ARGS.step_size, bp1=1 - (1 / ARGS.avg_window),
                        decay=ARGS.step_decay[0], power=ARGS.step_decay[1],
                        biased_g1=biased_g1)
                elif ARGS.optimizer == 'lbfgs':
                    self.optimizer = LBFGSOptimizer(self.model.img)
                else:
                    raise ValueError()

            params = self.model.img
            iters_i = ARGS.iterations[min(i, len(ARGS.iterations) - 1)]
            output_image = self.transfer(iters_i, params, content_scaled, style_scaled,
                                         callback, **kwargs)
            output_raw = self.current_raw

        return output_image


class Progress:
    """A helper class for keeping track of progress."""
    prev_t = None
    t = 0
    step = 0
    update_size = np.nan
    loss = np.nan
    tv_loss = np.nan

    def __init__(self, transfer, url=None, browser=None, steps=-1, save_every=0, web_if=None,
                 cli=None, callback=None):
        self.transfer = transfer
        self.url = url
        self.browser = browser
        self.steps = 0
        self.save_every = save_every
        self.web_if = web_if
        self.cli = cli
        self.callback = callback

    def __call__(self, step=-1, update_size=np.nan, loss=np.nan, tv_loss=np.nan, image=None):
        this_t = time.perf_counter()
        self.step += 1
        self.update_size = update_size
        self.loss = loss
        self.tv_loss = tv_loss
        if self.save_every and self.step % self.save_every == 0:
            self.transfer.current_output.save(RUN + '_out_%04d.png' % self.step)
        if self.step == 1:
            if self.url:
                if self.browser is None:
                    webbrowser.open(self.url)
                else:
                    webbrowser.get(self.browser).open(self.url)
            if self.cli:
                self.cli.start()
        else:
            self.t = this_t - self.prev_t
        print('Step %d, time: %.2f s, update: %.2f, loss: %e, tv: %.2f' %
               (step, self.t, update_size, loss, tv_loss), flush=True)
        self.web_if.put_event(
            web_interface.Iterate(self.step, self.steps, self.t, update_size, loss, tv_loss, image)
        )
        self.prev_t = this_t
        if self.callback:
            return self.callback()

    def set_steps(self, steps):
        self.steps = steps


def resize_to_fit(image, size, scale_up=False):
    """Resizes image to fit into a size-by-size square."""
    size = int(round(size)) // ARGS.div * ARGS.div
    w, h = image.size
    if not scale_up and max(w, h) <= size:
        return image
    new_w, new_h = w, h
    if w > h:
        new_w = size
        new_h = int(round(size * h / w)) // ARGS.div * ARGS.div
    else:
        new_h = size
        new_w = int(round(size * w / h)) // ARGS.div * ARGS.div
    return image.resize((new_w, new_h), Image.LANCZOS)


def printargs():
    """Prints out all command-line parameters."""
    bg = terminal_bg()
    if bg is True:
        style = 'xcode'
    elif bg is False:
        style = 'monokai'

    pprint = print
    try:
        if bg is not None:
            import pygments
            from pygments.lexers import Python3Lexer
            from pygments.formatters import Terminal256Formatter
            pprint = partial(pygments.highlight, lexer=Python3Lexer(),
                             formatter=Terminal256Formatter(style=style), outfile=sys.stdout)
    except ImportError:
        pass
    print('Parameters:')
    for key in sorted(ARGS):
        v = repr(getattr(ARGS, key))
        print('% 14s: ' % key, end='')
        pprint(v)
    print()


def get_image_comment():
    """Makes a comment string to write into the output image."""
    s = 'Created with https://github.com/crowsonkb/style_transfer.\n\n'
    s += 'Command line: style_transfer.py ' + ' '.join(sys.argv[1:]) + '\n\n'
    s += 'Parameters:\n'
    for item in sorted(vars(ARGS).items()):
        s += '%s: %s\n' % item
    return s


def init_model(resp_q, caffe_path, model, weights, mean):
    """Puts the list of layer shapes into resp_q. To be run in a separate process."""
    global logger
    setup_exceptions()
    logger = log_utils.setup_logger('init_model')

    if caffe_path:
        sys.path.append(caffe_path + '/python')
    import caffe
    caffe.set_mode_cpu()
    model = CaffeModel(model, weights, mean)
    shapes = {}
    for layer in model.layers():
        shapes[layer] = model.data[layer].shape
    resp_q.put(shapes)


VGG16_SHAPES = {
    'conv1_1': (64, 224, 224),
    'conv1_2': (64, 224, 224),
    'pool1': (64, 112, 112),
    'conv2_1': (128, 112, 112),
    'conv2_2': (128, 112, 112),
    'pool2': (128, 56, 56),
    'conv3_1': (256, 56, 56),
    'conv3_2': (256, 56, 56),
    'conv3_3': (256, 56, 56),
    'pool3': (256, 28, 28),
    'conv4_1': (512, 28, 28),
    'conv4_2': (512, 28, 28),
    'conv4_3': (512, 28, 28),
    'pool4': (512, 14, 14),
    'conv5_1': (512, 14, 14),
    'conv5_2': (512, 14, 14),
    'conv5_3': (512, 14, 14),
    'pool5': (512, 7, 7),
}

VGG19_SHAPES = {
    'conv1_1': (64, 224, 224),
    'conv1_2': (64, 224, 224),
    'pool1': (64, 112, 112),
    'conv2_1': (128, 112, 112),
    'conv2_2': (128, 112, 112),
    'pool2': (128, 56, 56),
    'conv3_1': (256, 56, 56),
    'conv3_2': (256, 56, 56),
    'conv3_3': (256, 56, 56),
    'conv3_4': (256, 56, 56),
    'pool3': (256, 28, 28),
    'conv4_1': (512, 28, 28),
    'conv4_2': (512, 28, 28),
    'conv4_3': (512, 28, 28),
    'conv4_4': (512, 28, 28),
    'pool4': (512, 14, 14),
    'conv5_1': (512, 14, 14),
    'conv5_2': (512, 14, 14),
    'conv5_3': (512, 14, 14),
    'conv5_4': (512, 14, 14),
    'pool5': (512, 7, 7),
}


def main():
    """CLI interface for style transfer."""
    global ARGS, RUN, STATS

    ARGS = parse_args(STATE)
    setup_exceptions()
    printargs()

    start_time = time.perf_counter()
    now = datetime.now()
    RUN = '%02d%02d%02d_%02d%02d%02d' % \
        (now.year % 100, now.month, now.day, now.hour, now.minute, now.second)
    STATS = StatLogger()
    print('Run %s started.' % RUN)

    if MKL_THREADS is not None:
        print('MKL detected, %d threads maximum.' % MKL_THREADS)

    os.environ['GLOG_minloglevel'] = '2'
    if ARGS.caffe_path:
        sys.path.append(ARGS.caffe_path + '/python')

    if Path(ARGS.model).name in ('vgg16.prototxt', 'vgg16_avgpool.prototxt'):
        shapes = VGG16_SHAPES
    elif Path(ARGS.model).name in ('vgg19.prototxt', 'vgg19_avgpool.prototxt'):
        shapes = VGG19_SHAPES
    else:
        print('Loading %s.' % ARGS.weights)
        resp_q = CTX.Queue()
        CTX.Process(target=init_model,
                    args=(resp_q, ARGS.caffe_path, ARGS.model, ARGS.weights, ARGS.mean)).start()
        shapes = resp_q.get()

    print('Initializing %s.' % ARGS.weights)
    model = CaffeModel(ARGS.model, ARGS.weights, ARGS.mean, shapes=shapes, placeholder=True)
    transfer = StyleTransfer(model)
    if ARGS.list_layers:
        print('Layers:')
        for layer, shape in model.shapes.items():
            print('% 25s %s' % (layer, shape))
        sys.exit(0)

    content_image = Image.open(ARGS.content_image).convert('RGB')
    style_images = []
    for image in ARGS.style_images:
        style_images.append(Image.open(image).convert('RGB'))
    initial_image, aux_image = None, None
    if ARGS.init_image:
        initial_image = Image.open(ARGS.init_image).convert('RGB')
    if ARGS.aux_image:
        aux_image = Image.open(ARGS.aux_image).convert('RGB')

    cli, cli_resp = None, None

    url = 'http://127.0.0.1:%d/' % ARGS.port
    progress_args = {}
    if ARGS.display == 'browser' and 'no_browser' not in ARGS:
        progress_args['url'] = url
        progress_args['browser'] = ARGS.browser
    steps = 0
    web_if = web_interface.WebInterface()
    th = threading.Thread(target=web_if.run, name='web_interface', args=(ARGS.port,), daemon=True)
    th.start()
    print('\nWatch the progress at: %s\n' % url)
    progress = Progress(
        transfer, steps=steps, save_every=ARGS.save_every, web_if=web_if, cli=cli,
        callback=cli_resp, **progress_args)

    np.random.seed(ARGS.seed)
    try:
        transfer.transfer_multiscale(
            [content_image], style_images, initial_image, aux_image, callback=progress)
        web_if.put_event(web_interface.IterationFinished())
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        STATS.dump()

    if transfer.current_output:
        output_image = ARGS.output_image
        if not output_image:
            output_image = RUN + '_out.png'
        print('Saving output as %s.' % output_image)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_itxt('Comment', get_image_comment())
        transfer.current_output.save(output_image, pnginfo=png_info)
    time_spent = time.perf_counter() - start_time
    print('Run %s ending after %dm %.3fs.' % (RUN, time_spent // 60, time_spent % 60))


if __name__ == '__main__':
    main()
