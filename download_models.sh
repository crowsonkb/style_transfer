#!/bin/sh

# Downloads the 19-layer VGG ILSVRC-2014 model (http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

curl -O 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt'

curl -O 'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
