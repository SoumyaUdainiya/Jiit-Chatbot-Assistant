# utils.py

import requests

def is_online():
    try:
        requests.get("https://www.google.com", timeout=2)
        return True
    except:
        return False