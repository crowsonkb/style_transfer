# style_transfer

Data-parallel neural style transfer using Caffe. Implements http://arxiv.org/abs/1508.06576.

Requirements:
- Python 3.5
- [Caffe](http://caffe.berkeleyvision.org), with pycaffe compiled for Python 3.5
- Python packages numpy, Pillow, posix_ipc, scipy

It will run faster if numpy is compiled to use [MKL](https://software.intel.com/en-us/intel-mkl). If you are running Caffe on the CPU, it will also be faster if compiled with MKL.

## Features

- The image is divided into tiles which are processed one at a time (with one GPU). Since the tiles can be sized so as to fit into GPU memory, this allows arbitrary size images to be processed&mdash;including print size. (ex: `--size 2048 --tile-size 512`)
- Images can be processed at multiple scales for speed. For instance, `--size 512 1024 2048 -i 100` will run 100 iterations (the default is 200) at 512x512, then 100 at 1024x1024, then 100 more at 2048x2048.
- Multi-GPU support (ex: `--devices 0 1 2 3`). Four GPUs, for instance, can process four tiles at a time.

## Example

The obligatory Golden Gate Bridge + Starry Night style transfer ([big version](https://s3-us-west-2.amazonaws.com/cb0a-46ef-cc86-8dda/style_transfer_examples/golden_gate_sn.jpg)):

<img src="examples/golden_gate_sn.jpg" width="512" height="384">
