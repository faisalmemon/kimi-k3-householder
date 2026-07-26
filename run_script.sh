#!/bin/bash

docker run --gpus all -it \
  --ipc=host \
  --net=host \
  --name kimi_debug \
  -v ~/dev/kimi-k3-householder:/workspace \
  kimi-householder:latest
