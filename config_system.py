"""The configuration system."""

import argparse
from fractions import Fraction
import math
import os
from pathlib import Path
import sys

import numpy as np

CONFIG_PY = Path(__file__).parent.resolve() / 'config.py'


def ffloat(s):
    """Parses fractional or floating point input strings."""
    return float(Fraction(s))


class arg:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

def add_args(parser, args):
    for a in args:
        parser.add_argument(*a.args, **a.kwargs)


def parse_args(state_obj=None):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser, [
        arg('--content-image', '-ci', help='the content image'),
        arg('--style-images', '-si', nargs='+', default=[], metavar='STYLE_IMAGE',
            help='the style images'),
        arg('--output_image', '-oi', help='the output image'),
        arg('--init-image', '-ii', metavar='IMAGE', help='the initial image'),
        arg('--aux-image', '-ai', metavar='IMAGE', help='the auxiliary image'),
        arg('--config', type=Path, help='a Python source file containing configuration options'),
        arg('--list-layers', action='store_true', help='list the model\'s layers'),
        arg('--caffe-path', help='the path to the Caffe installation'),
        arg('--devices', nargs='+', metavar='DEVICE', type=int, default=[-1],
            help='GPU device numbers to use (-1 for cpu)'),
        arg('--iterations', '-i', nargs='+', type=int, default=[200, 100],
            help='the number of iterations'),
        arg('--size', '-s', type=int, default=256, help='the output size'),
        arg('--min-size', type=int, default=182, help='the minimum scale\'s size'),
        arg('--style-scale', '-ss', type=ffloat, default=1, help='the style scale factor'),
        arg('--style-scale-up', default=False, action='store_true',
            help='allow scaling style images up'),
        arg('--tile-size', type=int, default=512, help='the maximum rendering tile size'),
        arg('--optimizer', '-o', default='adam', choices=['adam', 'lbfgs'],
            help='the optimizer to use'),
        arg('--step-size', '-st', type=ffloat, default=15,
            help='the initial step size for Adam'),
        arg('--step-decay', '-sd', nargs=2, metavar=('DECAY', 'POWER'), type=ffloat,
            default=[0.05, 0.5], help='on step i, divide step_size by (1 + DECAY * i)^POWER'),
        arg('--avg-window', type=ffloat, default=20, help='the iterate averaging window size'),
        arg('--layer-weights', help='a json file containing per-layer weight scaling factors'),
        arg('--content-weight', '-cw', type=ffloat, default=0.05, help='the content image factor'),
        arg('--dd-weight', '-dw', type=ffloat, default=0, help='the Deep Dream factor'),
        arg('--tv-weight', '-tw', type=ffloat, default=5, help='the TV smoothing factor'),
        arg('--tv-power', '-tp', metavar='BETA', type=ffloat, default=2,
            help='the TV smoothing exponent'),
        arg('--swt-weight', '-ww', metavar='WEIGHT', type=ffloat, default=0,
            help='the SWT smoothing factor'),
        arg('--swt-wavelet', '-wt', metavar='WAVELET', default='haar',
            help='the SWT wavelet'),
        arg('--swt-levels', '-wl', metavar='LEVELS', default=1, type=int,
            help='the number of levels to use for decomposition'),
        arg('--swt-power', '-wp', metavar='P', default=2, type=ffloat,
            help='the SWT smoothing exponent'),
        arg('--p-weight', '-pw', type=ffloat, default=2, help='the p-norm regularizer factor'),
        arg('--p-power', '-pp', metavar='P', type=ffloat, default=6, help='the p-norm exponent'),
        arg('--aux-weight', '-aw', type=ffloat, default=10, help='the auxiliary image factor'),
        arg('--content-layers', nargs='*', default=['conv4_2'],metavar='LAYER',
            help='the layers to use for content'),
        arg('--style-layers', nargs='*', metavar='LAYER',
            default=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
            help='the layers to use for style'),
        arg('--dd-layers', nargs='*', metavar='LAYER', default=[],
            help='the layers to use for Deep Dream'),
        arg('--port', '-p', type=int, default=8000, help='the port to use for the http server'),
        arg('--display', default='browser', choices=['browser', 'gui', 'none'],
            help='the display method to use'),
        arg('--hidpi', action='store_true', help='display the image at 2x scale in the browser'),
        arg('--prompt', action='store_true', help='enable the experimental prompt'),
        arg('--model', default='vgg19.prototxt',
            help='the Caffe deploy.prototxt for the model to use'),
        arg('--weights', default='vgg19.caffemodel',
            help='the Caffe .caffemodel for the model to use'),
        arg('--mean', nargs=3, metavar=('B_MEAN', 'G_MEAN', 'R_MEAN'),
            default=(103.939, 116.779, 123.68),
            help='the per-channel means of the model (BGR order)'),
        arg('--save-every', metavar='N', type=int, default=0, help='save the image every n steps'),
        arg('--seed', type=int, default=0, help='the random seed'),
        arg('--div', metavar='FACTOR', type=int, default=1,
            help='Ensure all images are divisible by FACTOR '
                 '(can fix some GPU memory alignment issues)'),
        arg('--jitter', action='store_true',
            help='use slower but higher quality translation-invariant rendering'),
        arg('--debug', action='store_true', help='enable debug messages'),
    ])

    defaults = vars(parser.parse_args([]))
    config_args = {}
    if CONFIG_PY.exists():
        config_args.update(eval_config(CONFIG_PY))
    sysv_args = vars(parser.parse_args())
    config2_args = {}
    if sysv_args['config']:
        config2_args.update(eval_config(sysv_args['config']))

    args = {}
    args.update(defaults)
    args.update(config_args)
    for a, value in sysv_args.items():
        if defaults[a] != value:
            args[a] = value
    args.update(config2_args)

    args2 = AutocallNamespace(state_obj, **args)
    if args2.debug:
        os.environ['DEBUG'] = '1'
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

    def __setattr__(self, name, value):
        if name in ('state_obj', 'ns'):
            object.__setattr__(self, name, value)
            return
        setattr(self.ns, name, value)

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
