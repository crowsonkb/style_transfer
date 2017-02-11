#!/bin/bash

if [[ ! -e style_transfer.py ]]; then
  echo 'Run this from within the style_transfer directory.' >&2
  exit 1
fi

if [[ ! -e .updated ]]; then
  touch .updated
  echo 'Running style_transfer init/update.'
  git pull
  if [[ ! -e vgg19.caffemodel ]]; then
    ./download_models.sh
  fi
fi
