"""The configuration system."""

import argparse
from fractions import Fraction
import math
from pathlib import Path
import sys

import numpy as np

CONFIG_PY = Path(__file__).parent.resolve() / 'config.py'


def ffloat(s):
    """Parses fractional or floating point input strings."""
    return float(Fraction(s))


def parse_args(state_obj=None):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content-image', '-ci', help='the content image')
    parser.add_argument('--style-images', '-si', nargs='+', default=[],
                        metavar='STYLE_IMAGE', help='the style images')
    parser.add_argument('--output_image', '-oi', default='out.png', help='the output image')
    parser.add_argument('--init-image', '-ii', metavar='IMAGE', help='the initial image')
    parser.add_argument('--aux-image', '-ai', metavar='IMAGE', help='the auxiliary image')
    parser.add_argument('--style-masks', nargs='+', metavar='MASK', default=[],
                        help='the masks for each style image')
    parser.add_argument('--config', default=CONFIG_PY, type=Path,
                        help='a Python source file containing configuration options')
    parser.add_argument('--list-layers', action='store_true', help='list the model\'s layers')
    parser.add_argument('--caffe-path', help='the path to the Caffe installation')
    parser.add_argument(
        '--devices', nargs='+', metavar='DEVICE', type=int, default=[-1],
        help='GPU device numbers to use (-1 for cpu)')
    parser.add_argument(
        '--iterations', '-i', nargs='+', type=int, default=[200, 100],
        help='the number of iterations')
    parser.add_argument(
        '--size', '-s', type=int, default=256, help='the output size')
    parser.add_argument(
        '--min-size', type=int, default=182, help='the minimum scale\'s size')
    parser.add_argument(
        '--style-scale', '-ss', type=ffloat, default=1, help='the style scale factor')
    parser.add_argument(
        '--style-scale-up', default=False, action='store_true',
        help='allow scaling style images up')
    parser.add_argument(
        '--tile-size', type=int, default=512, help='the maximum rendering tile size')
    parser.add_argument('--optimizer', '-o', default='adam', choices=['adam', 'lbfgs'],
                        help='the optimizer to use')
    parser.add_argument(
        '--step-size', '-st', type=ffloat, default=15,
        help='the initial step size for Adam')
    parser.add_argument(
        '--step-decay', '-sd', nargs=2, metavar=('GAMMA', 'POWER'), type=ffloat,
        default=[0.05, 0.5], help='on step i, divide step_size by (1 + GAMMA * i)^POWER')
    parser.add_argument(
        '--avg-window', type=ffloat, default=20, help='the iterate averaging window size')
    parser.add_argument(
        '--layer-weights', help='a json file containing per-layer weight scaling factors')
    parser.add_argument(
        '--content-weight', '-cw', type=ffloat, default=0.05, help='the content image factor')
    parser.add_argument(
        '--dd-weight', '-dw', type=ffloat, default=0, help='the Deep Dream factor')
    parser.add_argument(
        '--denoiser', '-d', default='tv', choices=['tv', 'wavelet'],
        help='the denoiser to use; \'wavelet\' is higher quality but consumes more CPU.')
    parser.add_argument(
        '--tv-weight', '-tw', type=ffloat, default=5, help='the TV smoothing factor')
    parser.add_argument(
        '--tv-power', '-tp', metavar='BETA', type=ffloat, default=2,
        help='the TV smoothing exponent')
    parser.add_argument(
        '--wt-type', '-wt', metavar='WAVELET', default='db4', help='the wavelet type to use')
    parser.add_argument(
        '--wt-weight', '-ww', type=ffloat, default=20, help='the wavelet smoothing factor')
    parser.add_argument(
        '--wt-power', '-wp', metavar='P', type=ffloat, default=3,
        help='the wavelet smoothing exponent')
    parser.add_argument(
        '--p-weight', '-pw', type=ffloat, default=2, help='the p-norm regularizer factor')
    parser.add_argument(
        '--p-power', '-pp', metavar='P', type=ffloat, default=6, help='the p-norm exponent')
    parser.add_argument(
        '--aux-weight', '-aw', type=ffloat, default=10, help='the auxiliary image factor')
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
        '--seed', type=int, default=0, help='the random seed')

    args = vars(parser.parse_args())
    args.update(eval_config(args['config']))
    args2 = AutocallNamespace(state_obj, **args)
    if not args2.list_layers and (not args2.content_image or not args2.style_images):
        parser.print_help()
        sys.exit(1)
    return args2


class ValuePlaceholder:
    pass


class AutocallNamespace:
    def __init__(self, state_obj, **kwargs):
        self.state_obj = state_obj
        self.ns = argparse.Namespace(**kwargs)

    def __getattr__(self, name):
        value = getattr(self.ns, name)
        if callable(value):
            try:
                return value(self.state_obj)
            except AttributeError:
                return ValuePlaceholder()
        return value

    def __iter__(self):
        yield from vars(self.ns)

    def __contains__(self, key):
        return key in self.ns

    def __repr__(self):
        return 'Autocall' + repr(self.ns)

CONFIG_GLOBALS = dict(math=math, np=np)


def eval_config(config_file):
    config_code = compile(config_file.read_text(), config_file.name, 'exec')
    locs = {}
    exec(config_code, CONFIG_GLOBALS, locs)  # pylint: disable=exec-used
    return locs
