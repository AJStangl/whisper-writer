import os
import sys
import subprocess
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Disabling output buffering so that the status window can be updated in real time
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
print('Starting WhisperWriter...')
subprocess.run([sys.executable, os.path.join('src', 'main.py')])
