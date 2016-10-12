#!/bin/sh

echo 'Downloading the 19-layer VGG ILSVRC-2014 model (http://www.robots.ox.ac.uk/~vgg/research/very_deep/).'

curl -O 'https://style-transfer.s3-us-west-2.amazonaws.com/VGG_ILSVRC_19_layers.caffemodel'
