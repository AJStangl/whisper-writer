import sys
import subprocess
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'

print('Starting WhisperWriter...')
subprocess.run([sys.executable, os.path.join('src', 'main.py')])
