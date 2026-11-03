#!/bin/bash
# Install JAX + optax in the WSL py313 env for the DFTHC trainer.
set -e
/home/kaggle/miniconda3/envs/py313/bin/python -m pip install --quiet "jax[cpu]" optax
/home/kaggle/miniconda3/envs/py313/bin/python -c "import jax, optax; print('jax', jax.__version__, 'backend', jax.default_backend(), 'optax ok')"
