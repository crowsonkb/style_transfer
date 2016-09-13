# style_transfer

Data-parallel neural style transfer using Caffe. Implements [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576).

Dependencies:
- [Python](https://www.python.org) 3.5
- [Caffe](http://caffe.berkeleyvision.org), with pycaffe compiled for Python 3.5
- Python packages [numpy](http://www.numpy.org), [Pillow](https://python-pillow.org), [posix-ipc](http://semanchuk.com/philip/posix_ipc/), [scipy](http://www.scipy.org)

`style_transfer` will run faster if numpy is compiled to use [MKL](https://software.intel.com/en-us/intel-mkl). If you are running Caffe on the CPU, it will also be faster if compiled with MKL. `style_transfer` uses one worker process per GPU, so the optimal value for `MKL_NUM_THREADS` is the number of real CPU cores divided by the number of GPUs.

## Features

- The image is divided into tiles which are processed one per GPU at a time. Since the tiles can be sized so as to fit into GPU memory, this allows arbitrary size images to be processed&mdash;including print size. (ex: `--size 2048 --tile-size 512`)
- Images can be processed at multiple scales for speed. For instance, `--size 512 1024 2048 -i 100` will run 100 iterations (the default is 200) at 512x512, then 100 at 1024x1024, then 100 more at 2048x2048.
- Multi-GPU support (ex: `--devices 0 1 2 3`). Four GPUs, for instance, can process four tiles at a time.
- Uses [Polyak-Ruppert averaging](https://www.researchgate.net/profile/Boris_Polyak2/publication/236736831_Acceleration_of_stochastic_approximation_by_averaging_SIAM_J_Control_Optim_30_838-855/links/0f31753227e964baab000000.pdf) over successive iterations to reduce image noise.


## Known issues

- Use of more than one content layer will produce incorrect feature maps when there is more than one tile.

## Example

The obligatory Golden Gate Bridge + Starry Night style transfer ([big version](https://s3-us-west-2.amazonaws.com/cb0a-46ef-cc86-8dda/style_transfer_examples/golden_gate_sn_big.jpg)):

<img src="examples/golden_gate_sn.jpg" width="512" height="384">

## Installation

### Building pycaffe for Python 3.5

*If you use pycaffe for other things, you might want to do this in a second copy of Caffe so you don't break things using Python 2.*

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

### Installing style_transfer's Python dependencies

First, install `style_transfer`'s direct dependencies:

```
pip3 install -Ur requirements.txt
```

Then, if you haven't already, Caffe's:

```
pip3 install -U scikit-image matplotlib
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
