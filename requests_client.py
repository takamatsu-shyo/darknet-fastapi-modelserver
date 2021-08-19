import requests
import cv2
import numpy as np
import sys

import cProfile
import pstats
import io
from pstats import SortKey

import time

# pr = cProfile.Profile()
# pr.enable()

# actual code for profiling

t1 = time.process_time()
img = cv2.resize(cv2.imread('images/frame4.jpg'), (416, 416))
img_encoded = cv2.imencode('.jpg', img)[1]
img_bytes = img_encoded.tobytes()  # bytes class
img_hex = img_bytes.hex()

r = requests.post('http://127.0.0.1:8000/predict/', json={"img": img_hex, "dim": (
    416, 416, 3), "model_id": 0, "save_predictions_on_server": True})
t2 = time.process_time()

print("Took ", round(t2-t1, 3), "sec")
print(r.content)
