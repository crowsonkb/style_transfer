#!/usr/bin/env python3

"""Generates smooth noise images."""

import argparse

import numpy as np
from PIL import Image

from num_utils import resize


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', '-o', default='noise.png', help='the output filename')
    parser.add_argument('--size', '-s', type=int, nargs=2, metavar=('W', 'H'),
                        default=[512, 512], help='the image size to output')
    parser.add_argument('--power', type=float, default=2, help='the bias toward large/small scales')
    parser.add_argument('--mean', type=float, default=0.5, help='the output mean')
    parser.add_argument('--std', type=float, default=0.1, help='the output standard deviation')
    args = parser.parse_args()

    scales = np.int32(max(np.ceil(np.log2(args.size)))) + 1
    for scale in range(scales):
        if scale == 0:
            canvas = np.zeros((1, 1), np.float32)
        else:
            canvas = resize(canvas, (2**scale, 2**scale))
        canvas += np.random.uniform(-1, 1, canvas.shape) / (scale+1)**2
    crop = canvas[:args.size[1], :args.size[0]]
    crop *= args.std / np.std(crop)
    crop += args.mean - np.mean(crop)

    Image.fromarray(np.uint8(np.clip(crop * 255, 0, 255))).save(args.output)

if __name__ == '__main__':
    main()
