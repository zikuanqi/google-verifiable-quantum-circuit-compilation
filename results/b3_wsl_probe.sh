#!/bin/bash
# Probe WSL python environments for jax/optax.
for e in /home/kaggle/miniconda3/envs/*/bin/python /home/kaggle/miniconda3/bin/python /usr/bin/python3; do
  if [ -x "$e" ]; then
    echo "== $e"
    "$e" -c 'import sys; print(sys.version.split()[0]); import jax; print("jax", jax.__version__)' 2>&1 | tail -2
  fi
done
