#!/usr/bin/env python3
"""Run Task 5 training in subprocess."""
import subprocess
import sys

# Run the training script
result = subprocess.run(
    [sys.executable, "task5_simclr_pretraining_fast.py"],
    cwd="/home/sadam/ITU/Deep_learning/DL_Assignment5_SimCLR_MSDS25069"
)
sys.exit(result.returncode)
