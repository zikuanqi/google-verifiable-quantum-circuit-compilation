#!/bin/bash
# WSL Ubuntu-24.04 setup for the Kaggle verifiable circuit compilation project.
set -e
export APT_OPTS="-o Acquire::ForceIPv4=true -o Acquire::http::Timeout=15 -o Acquire::Retries=2"
sudo apt-get update -qq $APT_OPTS
sudo apt-get install -y -qq -o Acquire::ForceIPv4=true python3-pip python3-venv python3-full
python3 -m venv /home/kaggle/venv
/home/kaggle/venv/bin/pip install --quiet numpy scipy
/home/kaggle/venv/bin/python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
