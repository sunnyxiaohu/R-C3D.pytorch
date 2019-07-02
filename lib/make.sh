#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

#python setup.py build_ext --inplace
#rm -rf build

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

# compile NMS
echo "Building nms op..."
cd ./model/nms
if [ -d "build" ]; then
    rm -r build
fi
python setup.py build_ext --inplace

