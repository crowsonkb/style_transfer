#!/usr/bin/env python3

"""Neural style transfer using Caffe. Implements http://arxiv.org/abs/1508.06576."""

# pylint: disable=invalid-name, too-many-arguments, too-many-locals

import argparse
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import os
from socketserver import ThreadingMixIn
import sys
import threading
import time
import webbrowser

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

os.environ['GLOG_minloglevel'] = '2'
import caffe  # pylint: disable=wrong-import-position

# Machine epsilon for float32
EPS = np.finfo(np.float32).eps


def normalize(arr):
    return arr / (np.mean(np.abs(arr)) + EPS)


def gram_matrix(feat):
    n, mh, mw = feat.shape
    feat = feat.reshape((n, mh * mw))
    gram = feat @ feat.T / feat.size
    return gram


class LayerIndexer:
    def __init__(self, net, attr):
        self.net, self.attr = net, attr

    def __getitem__(self, key):
        return getattr(self.net.blobs[key], self.attr)[0]

    def __setitem__(self, key, value):
        getattr(self.net.blobs[key], self.attr)[0] = value


class CaffeModel:
    def __init__(self, deploy, weights, mean=(0, 0, 0), bgr=True):
        self.mean = np.float32(mean)[..., None, None]
        assert self.mean.ndim == 3
        self.bgr = bgr
        self.net = caffe.Net(deploy, 1, weights=weights)
        self.data = LayerIndexer(self.net, 'data')
        self.diff = LayerIndexer(self.net, 'diff')

    def get_image(self):
        arr = self.data['data'] + self.mean
        if self.bgr:
            arr = arr[::-1]
        arr = arr.transpose((1, 2, 0))
        return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))

    def set_image(self, img):
        arr = np.float32(img).transpose((2, 0, 1))
        if self.bgr:
            arr = arr[::-1]
        self.net.blobs['data'].reshape(1, 3, *arr.shape[-2:])
        self.data['data'] = arr - self.mean

    def layers(self):
        """Returns the layer names of the network."""
        layers = []
        for i, layer in enumerate(self.net.blobs.keys()):
            if i == 0:
                continue
            if layer.find('_split_') == -1:
                layers.append(layer)
        return layers

    def transfer(self, iterations, content_image, style_image, content_layers, style_layers,
                 step_size=1, content_weight=1, style_weight=1, tv_weight=0, callback=None):
        b1, b2 = 0.9, 0.9

        content_weight /= max(len(content_layers), 1)
        style_weight /= max(len(style_layers), 1)

        # Construct list of layers to visit during the backward pass
        layers = []
        for layer in reversed(self.layers()):
            if layer in content_layers or layer in style_layers:
                layers.append(layer)

        # Prepare feature maps from content image
        features = {}
        self.set_image(content_image)
        self.net.forward(end=layers[0])
        for layer in content_layers:
            features[layer] = self.data[layer].copy()

        # Prepare Gram matrices from style image
        grams = {}
        self.set_image(style_image)
        self.net.forward(end=layers[0])
        for layer in style_layers:
            grams[layer] = gram_matrix(self.data[layer])

        # Initialize the model with a noise image
        w, h = content_image.size
        initial_image = np.random.uniform(0, 255, size=(h, w, 3))
        # initial_image *= np.std(style_image, axis=(0, 1), keepdims=True)
        # initial_image += np.mean(style_image, axis=(0, 1), keepdims=True)
        self.set_image(initial_image)
        m1 = np.zeros((3, h, w), dtype=np.float32)
        m2 = np.zeros((3, h, w), dtype=np.float32)

        for step in range(1, iterations+1):
            # Prepare gradient buffers and run the model forward
            for layer in layers:
                self.diff[layer] = 0
            self.net.forward(end=layers[0])

            for i, layer in enumerate(layers):
                # Compute the content and style gradients
                if layer in content_layers:
                    c_grad = self.data[layer] - features[layer]
                    self.diff[layer] += normalize(c_grad)*content_weight
                if layer in style_layers:
                    current_gram = gram_matrix(self.data[layer])
                    n, mh, mw = self.data[layer].shape
                    feat = self.data[layer].reshape((n, mh * mw))
                    s_grad = (feat.T @ (current_gram - grams[layer])).T
                    s_grad = s_grad.reshape((n, mh, mw))
                    self.diff[layer] += normalize(s_grad)*style_weight

                # Run the model backward
                if i+1 == len(layers):
                    self.net.backward(start=layer)
                else:
                    self.net.backward(start=layer, end=layers[i+1])

            # Compute total variation gradient
            tv_kernel = np.float32([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]])
            tv = convolve(self.data['data'], tv_kernel)/255

            # Compute a weighted sum of normalized gradients
            grad = normalize(self.diff['data']) + tv_weight*tv

            # Adam update
            m1 = b1*m1 + (1-b1)*grad
            m2 = b2*m2 + (1-b2)*grad**2
            update = step_size * m1/(1-b1**step) / (np.sqrt(m2/(1-b2**step)) + EPS)
            self.data['data'] -= update

            if callback is not None:
                callback(step=step, update_size=np.mean(np.abs(update)))

        return self.get_image()


class Progress:
    prev_t = None
    t = np.nan
    step = None

    def __init__(self, model, url=None, steps=-1):
        self.model = model
        self.url = url
        self.steps = steps
        self.update_size = np.nan

    def __call__(self, step=None, update_size=np.nan):
        this_t = time.perf_counter()
        self.step = step
        self.update_size = update_size
        if step == 1:
            if self.url:
                webbrowser.open(self.url)
        else:
            self.t = this_t - self.prev_t
        print('Step %d, time: %.2f s, update size: %.2f' % (step, self.t, update_size), flush=True)
        self.prev_t = this_t


class ProgressServer(ThreadingMixIn, HTTPServer):
    model = None
    progress = None


class ProgressHandler(BaseHTTPRequestHandler):
    index = """
    <meta http-equiv="refresh" content="5">
    <h1>style_transfer</h1>
    <img src="/out.png" width="%(w)d" height="%(h)d">
    <p>Step %(step)d/%(steps)d, time: %(t).2f s/step, update size: %(update_size).2f
    """

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write((self.index % {
                'step': self.server.progress.step,
                'steps': self.server.progress.steps,
                't': self.server.progress.t,
                'update_size': self.server.progress.update_size,
                'w': self.server.model.data['data'].shape[2],
                'h': self.server.model.data['data'].shape[1],
            }).encode())
        elif self.path == '/out.png':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            buf = io.BytesIO()
            self.server.model.get_image().save(buf, format='png')
            self.wfile.write(buf.getvalue())
        else:
            self.send_error(404)


def resize_to_fit(image, size):
    size = round(size)
    w, h = image.size
    new_w, new_h = w, h
    if w > h:
        new_w = size
        new_h = round(size * h/w)
    else:
        new_h = size
        new_w = round(size * w/h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def ffloat(s):
    return float(Fraction(s))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()
    parser.add_argument('content_image', help='the content image')
    parser.add_argument('style_image', help='the style image')
    parser.add_argument('output_image', nargs='?', default='out.png', help='the output image')
    parser.add_argument(
        '-i', dest='iterations', type=int, default=500, help='the number of iterations')
    parser.add_argument(
        '-s', dest='step_size', type=ffloat, default=10,
        help='the step size (iteration strength)')
    parser.add_argument('--size', type=int, default=224, help='the maximum output size')
    parser.add_argument('--style-scale', type=ffloat, default=1, help='the style scale factor')
    parser.add_argument(
        '-cw', dest='content_weight', type=ffloat, default=0.1, help='the content image factor')
    parser.add_argument(
        '-tw', dest='tv_weight', type=ffloat, default=1, help='the smoothing factor')
    parser.add_argument(
        '--content-layers', nargs='*', default=['conv4_2'], metavar='LAYER',
        help='the layers to use for content')
    parser.add_argument(
        '--style-layers', nargs='*', metavar='LAYER',
        default=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        help='the layers to use for style')
    parser.add_argument(
        '-p', dest='port', type=int, default=8000, help='the port to use for the http server')
    parser.add_argument('--no-browser', action='store_true', help='don\'t open a web browser')
    parser.add_argument(
        '--model', default='VGG_ILSVRC_19_layers_deploy.prototxt',
        help='the Caffe deploy.prototxt for the model to use')
    parser.add_argument(
        '--model-weights', default='VGG_ILSVRC_19_layers.caffemodel',
        help='the Caffe .caffemodel for the model to use')
    parser.add_argument(
        '--model-mean', nargs=3, metavar=('B_MEAN', 'G_MEAN', 'R_MEAN'),
        default=(103.939, 116.779, 123.68),
        help='the per-channel means of the model (BGR order)')
    parser.add_argument(
        '--list-layers', action='store_true', help='list the model\'s layers')
    return parser.parse_args()


def main():
    args = parse_args()

    model = CaffeModel(args.model, args.model_weights, args.model_mean)
    if args.list_layers:
        print('Layers:')
        for layer in model.layers():
            print('    %s' % layer)
        sys.exit(0)

    content_image = Image.open(args.content_image).convert('RGB')
    style_image = Image.open(args.style_image).convert('RGB')
    content_image = resize_to_fit(content_image, args.size)
    style_image = resize_to_fit(style_image, args.size*args.style_scale)
    print('Resized content image to %dx%d' % content_image.size)
    print('Resized style image to %dx%d' % style_image.size)

    server_address = ('', args.port)
    url = 'http://127.0.0.1:%d/' % args.port
    server = ProgressServer(server_address, ProgressHandler)
    server.model = model
    progress_args = {}
    if not args.no_browser:
        progress_args['url'] = url
    server.progress = Progress(model, steps=args.iterations, **progress_args)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print('\nWatch the progress at: %s\n' % url)

    caffe.set_mode_gpu()  # FIXME: add an argument
    caffe.set_random_seed(0)
    np.random.seed(0)
    try:
        output_image = model.transfer(
            args.iterations, content_image, style_image, args.content_layers, args.style_layers,
            step_size=args.step_size, content_weight=args.content_weight, tv_weight=args.tv_weight,
            callback=server.progress)
    except KeyboardInterrupt:
        output_image = model.get_image()
    print('Saving output as %s.' % args.output_image)
    output_image.save(args.output_image)

if __name__ == '__main__':
    main()
