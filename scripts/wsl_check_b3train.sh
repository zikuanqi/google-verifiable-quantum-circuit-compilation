#!/bin/bash
ps aux | grep b3_train | grep -v grep | head -3
echo "--- log size ---"
ls -la /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b3_train_65_60_30.log
echo "--- log tail ---"
tail -4 /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b3_train_65_60_30.log
