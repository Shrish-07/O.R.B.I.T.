import subprocess, sys, time, urllib.request
from pathlib import Path

proc = subprocess.Popen(
    [sys.executable, '-m', 'streamlit', 'run', 'app/app.py',
     '--server.headless', 'true', '--server.port', '8504'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(15)
try:
    r = urllib.request.urlopen('http://localhost:8504', timeout=10)
    print('SMOKE TEST: PASS — status', r.status)
except Exception as e:
    out, err = proc.stdout.read(), proc.stderr.read()
    print('SMOKE TEST: FAIL —', e)
    print('STDERR:', err.decode('utf-8', errors='replace')[:3000])
finally:
    proc.terminate()
