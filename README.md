# style_transfer

Data-parallel image stylization using Caffe. Implements [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) [1].

Dependencies:
- [Python](https://www.python.org) 2.7 or 3.5
- [Caffe](http://caffe.berkeleyvision.org), with pycaffe compiled for Python 2.7 or 3.5
- Python packages [numpy](http://www.numpy.org), [Pillow](https://python-pillow.org), [posix-ipc](http://semanchuk.com/philip/posix_ipc/), [scipy](http://www.scipy.org), [six](https://pythonhosted.org/six/)

`style_transfer` will run faster if numpy is compiled to use [MKL](https://software.intel.com/en-us/intel-mkl). If you are running Caffe on the CPU, it will also be faster if compiled with MKL. `style_transfer` uses one worker process per GPU, so the optimal value for `MKL_NUM_THREADS` is the number of real CPU cores divided by the number of GPUs.

[Cloud computing images](https://github.com/crowsonkb/style_transfer/wiki/Cloud-computing-images) are available with `style_transfer` and its dependencies preinstalled.

## Features

- The image is divided into tiles which are processed one per GPU at a time. Since the tiles can be sized so as to fit into GPU memory, this allows arbitrary size images to be processed&mdash;including print size. Tile seam suppression is applied after every iteration so that seams do not accumulate and become visible. (ex: `--size 2048 --tile-size 1024`)
- Images can be processed at multiple scales. For instance, `--size 512 768 1024 1536 2048 -i 100` will run 100 iterations at 512x512, then 100 at 768x768, then 100 more at 1024x1024 etc. Each scale's final iterate is used as the initial iterate for the following scale. Processing a large image at smaller scales first markedly improves output quality.
- Multi-GPU support (ex: `--devices 0 1 2 3`). Four GPUs, for instance, can process four tiles at a time.
- Averages successive iterates [2] to reduce image noise.

## Known issues

- Use of more than one content layer will produce incorrect feature maps when there is more than one tile.

## Examples

The obligatory [Golden Gate Bridge](https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/golden_gate.jpg) + [The Starry Night](https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night.jpg) (van Gogh) style transfer ([big version](https://s3-us-west-2.amazonaws.com/cb0a-46ef-cc86-8dda/style_transfer_examples/golden_gate_sn_big.jpg)):

<img src="examples/golden_gate_sn.jpg" width="512" height="384">

[Golden Gate Bridge](https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/golden_gate.jpg) + [The Shipwreck of the Minotaur](https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/shipwreck.jpg) (Turner) ([big version](https://s3-us-west-2.amazonaws.com/cb0a-46ef-cc86-8dda/style_transfer_examples/golden_shipwreck.jpg)):

<img src="examples/golden_shipwreck.jpg" width="512" height="384">

[barn and pond](http://r0k.us/graphics/kodak/kodim22.html) (Cindy Branham) + [The Banks of the River](https://raw.githubusercontent.com/DmitryUlyanov/fast-neural-doodle/master/data/Renoir/style.png) (Renoir) ([big version](http://cb0a-46ef-cc86-8dda.s3.amazonaws.com/style_transfer_examples/kodim22_renoir.jpg)):

<img src="examples/kodim22_renoir.jpg" width="512" height="341.5">

## Installation

*Python 2 support is relatively untested. Please feel free to report issues! Be sure to specify which Python version you are using if you report an issue.*

*If you use pycaffe for other things, you might want to build pycaffe for Python 3 in a second copy of Caffe so you don't break things using Python 2.*

### Building pycaffe for Python 3.5 (OS X)

On OS X, you can install Python 3 and Boost.Python using [Homebrew](http://brew.sh):

```
brew install python3
brew install boost-python --with-python3
```

Then insert these lines into Caffe's `Makefile.config` to build against the Homebrew-provided Python 3.5:

```
PYTHON_DIR := /usr/local/opt/python3/Frameworks/Python.framework/Versions/3.5
PYTHON_LIBRARIES := boost_python3 python3.5m
PYTHON_INCLUDE := $(PYTHON_DIR)/include/python3.5m \
	/usr/local/lib/python3.5/site-packages/numpy/core/include
PYTHON_LIB := $(PYTHON_DIR)/lib
```

`make pycaffe` ought to compile the Python 3 bindings now.

### Building pycaffe for Python 3.5 (Ubuntu 16.04)

On Ubuntu 16.04, follow Caffe's [Ubuntu 15.10/16.04 install guide](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide). The required `Makefile.config` lines for Python 3.5 are:

```
PYTHON_LIBRARIES := boost_python-py35 python3.5m
PYTHON_INCLUDE := /usr/include/python3.5m \
                  /usr/local/lib/python3.5/dist-packages/numpy/core/include
PYTHON_LIB := /usr/lib
```

`style_transfer` is also known to work on Ubuntu 14.04 with Python 3.5 built from source.

### Installing style_transfer's Python dependencies

First, install `style_transfer`'s direct dependencies:

```
pip3 install -Ur requirements.txt
```

Then, if you haven't already, Caffe's:

```
pip3 install -U matplotlib scikit-image
```

Besides those, Caffe depends on [protobuf](https://github.com/google/protobuf), and versions of protobuf older than 3.0 do not work with Python 3&mdash;so go to the [releases page](https://github.com/google/protobuf/releases) and download the Python runtime library (example given is for version 3.0.2):

```
unzip protobuf-python-3.0.2.zip
cd protobuf-3.0.2/python
python3 setup.py bdist_wheel --cpp_implementation
pip3 install -U dist/*.whl
```

You should be able to run `style_transfer` now by specifying the path to Caffe's Python bindings in the `PYTHONPATH`:

```
PYTHONPATH="/path/to/caffe/python" python3 style_transfer.py <content_image> <style_image>
```

## References

[1] L. Gatys, A. Ecker, M. Bethge, "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)"

[2] O. Shamir, T. Zhang, "[Stochastic Gradient Descent for Non-smooth Optimization: Convergence Results and Optimal Averaging Schemes](http://jmlr.csail.mit.edu/proceedings/papers/v28/shamir13.pdf)"

[3] A. Mahendran, A. Vedaldi, "[Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035)"

[4] D. Kingma, J. Ba, "[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)"

[5] K. Simonyan, A. Zisserman, "[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)"
