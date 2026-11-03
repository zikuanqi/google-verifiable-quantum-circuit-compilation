#!/bin/bash
set -e
echo "=== symbol in libpython ==="
nm -D /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 | grep PyObject_ClearManagedDict || echo "NOT FOUND"
echo "=== ctypes check ==="
/usr/bin/python3.12 -c "import ctypes; print(ctypes.pythonapi.PyObject_ClearManagedDict)"
echo "=== import kickmix with system python ==="
/usr/bin/python3.12 -c "import sys; sys.path.insert(0, '/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix'); import kickmix; print('kickmix OK')"
