import requests
import cv2
import numpy as np
import sys
import cProfile
import pstats
import io
from pstats import SortKey
import time

mode='hex'
#mode='base64'
#mode='noncomp'

t1 = time.time()
img = cv2.resize(cv2.imread('images/frame4.jpg'), (416, 416))

if mode == 'base64':
    img_encoded = cv2.imencode('.jpg', img)[1]
    img_bytes = img_encoded.tobytes()
    import base64
    img_b64 = base64.b64encode(img_bytes)
    img_b64_str = img_b64.decode('ascii')
    print('len base64', len(img_b64_str))
    r = requests.post('http://127.0.0.1:8006/predict_b64/', json={"img": img_b64_str, "dim": (
        416, 416, 3), "model_id": 0, "save_predictions_on_server": False})

elif mode == 'noncomp':
    img_bytes = img.tobytes()
    img_hex = img_bytes.hex()
    print('len umcompressed', len(img_hex))
    r = requests.post('http://127.0.0.1:8006/predict_uncomp/', json={"img": img_hex, "dim": (
        416, 416, 3), "model_id": 0, "save_predictions_on_server": False})

else: # hex is default and fast
    img_encoded = cv2.imencode('.jpg', img)[1]
    print('len jpg', len(img_encoded))
    img_bytes = img_encoded.tobytes()  # bytes class
    img_hex = img_bytes.hex()
    print('len hex', len(img_hex))
    r = requests.post('http://127.0.0.1:8006/predict/', json={"img": img_hex, "dim": (
        416, 416, 3), "model_id": 0, "save_predictions_on_server": False})

t2 = time.time()
print("Took ", round(t2-t1, 3), "sec")
print(r.content)







