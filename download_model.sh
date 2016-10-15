#!/bin/sh

echo 'Downloading the VGG ILSVRC-2014 models (http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [5].'

curl -O 'https://style-transfer.s3-us-west-2.amazonaws.com/vgg16.caffemodel'
curl -O 'https://style-transfer.s3-us-west-2.amazonaws.com/vgg19.caffemodel'
