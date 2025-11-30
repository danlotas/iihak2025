import subprocess
import sys

# Запускаем streamlit через subprocess, который работает одинаково на Windows/Mac/Linux
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
