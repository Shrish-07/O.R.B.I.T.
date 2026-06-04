import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:8502', timeout=10)
    print('App responded:', r.status)
except Exception as e:
    print('App failed:', e)
