#!/bin/bash

HOST_WORKSPACE="${HOST_WORKSPACE:-$HOME/dev/kimi-k3-householder}"

docker run --gpus all -it \
  --ipc=host \
  --net=host \
  --name kimi_debug \
  -v "${HOST_WORKSPACE}":/workspace \
  kimi-householder:latest
