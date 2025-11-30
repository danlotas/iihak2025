import subprocess
import sys
import os

file = os.path.join(os.path.dirname(__file__), "app.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", file])
