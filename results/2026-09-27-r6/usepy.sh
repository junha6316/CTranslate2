#!/bin/bash
# usepy.sh TREE: switch the editable python package (and its compiled extension) to ~/w/TREE.
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/venv/bin/activate
export CTRANSLATE2_ROOT=$HOME/w/$1/install LD_LIBRARY_PATH=$HOME/w/$1/install/lib:/usr/local/cuda-12.8/lib64
pip install -q -e ~/w/$1/python --force-reinstall --no-deps 2>&1 | tail -2
cd /tmp && python -c 'import ctranslate2; print("ctranslate2.__file__ =", ctranslate2.__file__)'
