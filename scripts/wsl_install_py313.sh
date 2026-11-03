#!/bin/bash
# Create python 3.13 env via conda-forge only.
set -e
export PATH=/home/kaggle/miniconda3/bin:$PATH
conda config --set always_yes yes --set changeps1 no
conda config --remove channels defaults 2>/dev/null || true
conda config --remove-key channels 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority strict
conda config --show channels
if ! conda env list | grep -q "py313"; then
    conda create -n py313 -c conda-forge --override-channels python=3.13 -y -q
fi
/home/kaggle/miniconda3/envs/py313/bin/python -m pip install --quiet numpy scipy
/home/kaggle/miniconda3/envs/py313/bin/python -c "import sys, numpy, scipy; print(sys.version.split()[0]); print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
