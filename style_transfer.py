#!/usr/bin/env python3

"""Neural style transfer using Caffe. Implements http://arxiv.org/abs/1508.06576."""

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, too-many-locals

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
from scipy.ndimage import convolve, convolve1d

# Machine epsilon for float32
EPS = np.finfo(np.float32).eps


def normalize(arr):
    """Normalizes an array to have mean 1."""
    return arr / (np.mean(np.abs(arr)) + EPS)


def gram_matrix(feat):
    """Computes the Gram matrix corresponding to a feature map."""
    n, mh, mw = feat.shape
    feat = feat.reshape((n, mh * mw))
    gram = feat @ feat.T / feat.size
    return gram


class LayerIndexer:
    """Helper class for accessing feature maps and gradients."""
    def __init__(self, net, attr):
        self.net, self.attr = net, attr

    def __getitem__(self, key):
        return getattr(self.net.blobs[key], self.attr)[0]

    def __setitem__(self, key, value):
        getattr(self.net.blobs[key], self.attr)[0] = value


class Optimizer:
    """Implements the RMSprop gradient descent optimizer with Nesterov momentum."""
    def __init__(self, params, step_size=1, max_step=0, b1=0.9, b2=0.9):
        """Initializes the optimizer."""
        self.params = params
        self.step_size = step_size
        self.max_step = max_step
        self.b1 = b1
        self.b2 = b2
        self.step = 1
        self.m1 = np.zeros_like(params)
        self.m2 = np.zeros_like(params)

    def get_ss(self):
        """Get the current step size."""
        if self.max_step:
            # gamma = decay_by**(1/every)
            gamma = 0.05**(1/self.max_step)
            return self.step_size * gamma**self.step
        return self.step_size

    def update(self, grad, old_params):
        """Returns a step's parameter update given its gradient and pre-Nesterov-step params."""
        self.m1[:] = self.b1*self.m1 + grad
        self.m2[:] = self.b2*self.m2 + (1-self.b2)*grad**2
        update = self.get_ss() * self.m1 / (np.sqrt(self.m2) + EPS)

        self.params[:] = old_params - update
        self.step += 1
        return update

    def apply_nesterov_step(self):
        """Updates params with an estimate of the next update."""
        old_params = self.params.copy()
        self.params -= self.get_ss() * self.b1*self.m1 / (np.sqrt(self.b2*self.m2) + EPS)
        return old_params

    def roll(self, xy):
        x, y = xy
        self.m1[:] = np.roll(np.roll(self.m1, x, 2), y, 1)
        self.m2[:] = np.roll(np.roll(self.m2, x, 2), y, 1)


class CaffeModel:
    """A Caffe neural network model."""
    def __init__(self, deploy, weights, mean=(0, 0, 0), bgr=True):
        import caffe
        self.mean = np.float32(mean)[..., None, None]
        assert self.mean.ndim == 3
        self.bgr = bgr
        self.net = caffe.Net(deploy, 1, weights=weights)
        self.data = LayerIndexer(self.net, 'data')
        self.diff = LayerIndexer(self.net, 'diff')
        self.features = None
        self.grams = None
        self.current_output = None
        self.img = None

    def get_image(self):
        """Gets the current model input as a PIL image."""
        arr = self.img + self.mean
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

    def preprocess_images(self, content_image, style_image, content_layers, style_layers):
        """Performs preprocessing tasks on the input images."""
        # Construct list of layers to visit during the backward pass
        layers = []
        for layer in reversed(self.layers()):
            if layer in content_layers or layer in style_layers:
                layers.append(layer)

        # Prepare feature maps from content image
        self.features = {}
        self.set_image(content_image)
        self.net.blobs['data'].reshape(1, 3, *self.img.shape[-2:])
        self.data['data'] = self.img
        self.net.forward(end=layers[0])
        for layer in content_layers:
            self.features[layer] = self.data[layer].copy()

        # Prepare Gram matrices from style image
        if self.grams is None:
            self.grams = {}
            self.set_image(style_image)
            self.net.blobs['data'].reshape(1, 3, *self.img.shape[-2:])
            self.data['data'] = self.img
            self.net.forward(end=layers[0])
            for layer in style_layers:
                self.grams[layer] = gram_matrix(self.data[layer])

        return layers

    def eval_sc_grad_tile(self, img, start, layers, content_layers, style_layers,
                          content_weight, style_weight):
        self.net.blobs['data'].reshape(1, 3, *img.shape[-2:])
        self.data['data'] = img

        # Prepare gradient buffers and run the model forward
        for layer in layers:
            self.diff[layer] = 0
        self.net.forward(end=layers[0])
        # content_loss, style_loss = 0, 0

        for i, layer in enumerate(layers):
            # Compute the content and style gradients
            if layer in content_layers:
                # FIXME: adapt to different content layer sizes
                start = start // 8
                end = start + np.array(self.data[layer].shape[-2:])
                feat = self.features[layer][:, start[0]:end[0], start[1]:end[1]]
                c_grad = self.data[layer] - feat
                self.diff[layer] += normalize(c_grad)*content_weight
                # content_loss += 0.5 * np.sum((self.data[layer] - self.features[layer])**2)
            if layer in style_layers:
                current_gram = gram_matrix(self.data[layer])
                n, mh, mw = self.data[layer].shape
                feat = self.data[layer].reshape((n, mh * mw))
                s_grad = (current_gram - self.grams[layer]).T @ feat
                s_grad = s_grad.reshape((n, mh, mw))
                self.diff[layer] += normalize(s_grad)*style_weight
                # style_loss += 0.5 * np.sum((current_gram - self.grams[layer])**2)

            # Run the model backward
            if i+1 == len(layers):
                self.net.backward(start=layer)
            else:
                self.net.backward(start=layer, end=layers[i+1])

        return self.diff['data']

    def eval_sc_grad(self, *args, tile_size=256):
        grad = np.zeros_like(self.img)
        img_size = np.array(self.img.shape[-2:])
        ntiles = img_size // tile_size + 1
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
                grad_tile = self.eval_sc_grad_tile(tile, start, *args)
                grad[:, start[0]:end[0], start[1]:end[1]] = grad_tile

        return grad

    def roll(self, xy):
        """Roll image and feature maps. VGG only."""
        for layer, feat in self.features.items():
            assert layer.startswith('conv') or layer.startswith('pool')
            level = int(layer[4])
            if layer.startswith('pool'):
                level += 1
            assert level <= 4
            x, y = xy * 2**(4-level)
            self.features[layer] = np.roll(np.roll(feat, x, 2), y, 1)
        x, y = xy * 8
        self.img[:] = np.roll(np.roll(self.img, x, 2), y, 1)

    def transfer(self, iterations, content_image, style_image, content_layers, style_layers,
                 step_size=1, content_weight=1, style_weight=1, tv_weight=1, callback=None,
                 initial_image=None):
        """Performs style transfer from style_image to content_image."""
        content_weight /= max(len(content_layers), 1)
        style_weight /= max(len(style_layers), 1)

        layers = self.preprocess_images(content_image, style_image, content_layers, style_layers)

        # Initialize the model with a noise image
        w, h = content_image.size
        if initial_image is not None:
            assert initial_image.size == (w, h), 'Initial image size must match content image'
            self.set_image(initial_image)
        else:
            self.set_image(np.random.uniform(0, 255, size=(h, w, 3)))

        optimizer = Optimizer(self.img, step_size=step_size, max_step=iterations+1)
        log = open('log.csv', 'w')
        print('tv loss', file=log, flush=True)

        for step in range(1, iterations+1):
            xy = np.int32(np.random.uniform(0, 16, size=2))-8
            self.roll(xy)
            optimizer.roll(xy*8)

            old_params = optimizer.apply_nesterov_step()
            grad = self.eval_sc_grad(layers, content_layers, style_layers,
                                     content_weight, style_weight)

            # Compute total variation gradient
            tv_kernel = np.float32([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]])
            tv_grad = convolve(self.img, tv_kernel, mode='wrap')/255
            tv_mask = np.ones_like(tv_grad)
            tv_mask[:, :2, :] = 5
            tv_mask[:, -2:, :] = 5
            tv_mask[:, :, :2] = 5
            tv_mask[:, :, -2:] = 5
            tv_grad *= tv_mask

            # Compute a weighted sum of normalized gradients
            grad = normalize(grad) + tv_weight*tv_grad

            # In-place gradient descent update
            update_size = np.mean(np.abs(optimizer.update(grad, old_params)))
            self.roll(-xy)
            optimizer.roll(-xy*8)

            # Apply constraints
            mean = self.mean.squeeze()
            self.img[0] = np.clip(self.img[0], -mean[0], 255-mean[0])
            self.img[1] = np.clip(self.img[1], -mean[1], 255-mean[1])
            self.img[2] = np.clip(self.img[2], -mean[2], 255-mean[2])
            # print('mean params=%g' % np.mean(self.img))

            # Compute tv loss statistic
            tv_h = convolve1d(self.img, [-1, 1], axis=1)
            tv_v = convolve1d(self.img, [-1, 1], axis=2)
            tv_loss = 0.5 * np.sum((tv_h**2 + tv_v**2)) / self.img.size
            print(tv_loss, file=log, flush=True)

            self.current_output = self.get_image()

            if callback is not None:
                callback(step=step, update_size=update_size, loss=tv_loss)

        return self.get_image()

    def transfer_multiscale(self, sizes, iterations, content_image, style_image, *args, **kwargs):
        """Performs style transfer from style_image to content_image at the given sizes."""
        output_image = None
        for size in sizes:
            content_scaled = resize_to_fit(content_image, size)
            # style_scaled = resize_to_fit(style_image, size)
            if output_image is not None:
                output_image = output_image.resize(content_scaled.size, Image.BICUBIC)
            output_image = self.transfer(iterations, content_scaled, style_image,
                                         initial_image=output_image, *args, **kwargs)
        return output_image


class Progress:
    """A helper class for keeping track of progress."""
    prev_t = None
    t = 0
    step = 0
    update_size = np.nan
    loss = np.nan

    def __init__(self, model, url=None, steps=-1, save_every=0):
        self.model = model
        self.url = url
        self.steps = steps
        self.save_every = save_every

    def __call__(self, step=-1, update_size=np.nan, loss=np.nan):
        this_t = time.perf_counter()
        self.step += 1
        self.update_size = update_size
        self.loss = loss
        if self.save_every and self.step % self.save_every == 0:
            self.model.current_output.save('out_%04d.png' % self.step)
        if self.step == 1:
            if self.url:
                webbrowser.open(self.url)
        else:
            self.t = this_t - self.prev_t
        print('Step %d, time: %.2f s, mean update: %.2f, mean tv loss: %.1f' %
              (step, self.t, update_size, loss), flush=True)
        self.prev_t = this_t


class ProgressServer(ThreadingMixIn, HTTPServer):
    """HTTP server class."""
    model = None
    progress = None


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
            self.wfile.write((self.index % {
                'step': self.server.progress.step,
                'steps': self.server.progress.steps,
                't': self.server.progress.t,
                'update_size': self.server.progress.update_size,
                'loss': self.server.progress.loss,
                'w': self.server.model.current_output.size[0],
                'h': self.server.model.current_output.size[1],
            }).encode())
        elif self.path == '/out.png':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            buf = io.BytesIO()
            self.server.model.current_output.save(buf, format='png')
            self.wfile.write(buf.getvalue())
        else:
            self.send_error(404)


def resize_to_fit(image, size):
    """Resizes image to fit into a size-by-size square."""
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
    """Parses fractional or floating point input strings."""
    return float(Fraction(s))


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('content_image', help='the content image')
    parser.add_argument('style_image', help='the style image')
    parser.add_argument('output_image', nargs='?', default='out.png', help='the output image')
    parser.add_argument(
        '--iterations', '-i', type=int, default=200, help='the number of iterations')
    parser.add_argument(
        '--step-size', '-st', type=ffloat, default=2, help='the step size (iteration strength)')
    parser.add_argument(
        '--size', '-s', nargs='+', type=int, default=[256], help='the output size(s)')
    parser.add_argument(
        '--style-scale', '-ss', type=ffloat, default=1, help='the style scale factor')
    parser.add_argument(
        '--content-weight', '-cw', type=ffloat, default=0.05, help='the content image factor')
    parser.add_argument(
        '--tv-weight', '-tw', type=ffloat, default=1, help='the smoothing factor')
    parser.add_argument(
        '--content-layers', nargs='*', default=['conv4_2'], metavar='LAYER',
        help='the layers to use for content')
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
        '--save-every', metavar='N', type=int, default=0, help='save the image every n steps'
    )
    parser.add_argument(
        '--gpu', type=int, default=0, help='gpu number to use (-1 for cpu)'
    )
    return parser.parse_args()


def main():
    """CLI interface for style transfer."""
    args = parse_args()

    os.environ['GLOG_minloglevel'] = '2'
    if args.gpu == -1:
        import caffe
        caffe.set_mode_cpu()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        import caffe
        caffe.set_mode_gpu()

    model = CaffeModel(args.model, args.weights, args.mean)
    if args.list_layers:
        print('Layers:')
        for layer in model.layers():
            print('    %s, size=%s' % (layer, model.data[layer].shape))
        sys.exit(0)

    sizes = sorted(args.size)
    content_image = Image.open(args.content_image).convert('RGB')
    style_image = Image.open(args.style_image).convert('RGB')
    style_image = resize_to_fit(style_image, sizes[-1]*args.style_scale)
    print('Resized style image to %dx%d' % style_image.size)

    server_address = ('', args.port)
    url = 'http://127.0.0.1:%d/' % args.port
    server = ProgressServer(server_address, ProgressHandler)
    server.model = model
    progress_args = {}
    if not args.no_browser:
        progress_args['url'] = url
    server.progress = Progress(
        model, steps=args.iterations*len(sizes), save_every=args.save_every, **progress_args)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print('\nWatch the progress at: %s\n' % url)

    caffe.set_random_seed(0)
    np.random.seed(0)
    try:
        output_image = model.transfer_multiscale(
            sizes, args.iterations, content_image, style_image, args.content_layers,
            args.style_layers, step_size=args.step_size, content_weight=args.content_weight,
            tv_weight=args.tv_weight, callback=server.progress)
    except KeyboardInterrupt:
        output_image = model.get_image()
    print('Saving output as %s.' % args.output_image)
    output_image.save(args.output_image)

if __name__ == '__main__':
    main()
