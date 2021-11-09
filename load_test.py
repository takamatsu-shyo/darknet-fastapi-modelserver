import requests
import cv2
import numpy as np
import sys
import cProfile
import pstats
import io
from pstats import SortKey
import time
from multiprocessing import Process


def yolo_request(img_hex, model_id):
    r = requests.post('http://127.0.0.1:8006/predict/', json={"img": img_hex, "dim": (
        416, 416, 3), "model_id": model_id, "save_predictions_on_server": False})

    return r.content

def yr_100(img_hex, model_id):
    for i in range(100):
        return_content = yolo_request(img_hex, model_id)
 
def main():
    mode='hex'
    
    img = cv2.resize(cv2.imread('images/frame4.jpg'), (416, 416))
    img_encoded = cv2.imencode('.jpg', img)[1]
    img_bytes = img_encoded.tobytes()  # bytes class
    img_hex = img_bytes.hex()
 
    t1 = time.time()

    p1 = Process(target=yr_100, args=(img_hex,0,))
    p2 = Process(target=yr_100, args=(img_hex,1,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    t2 = time.time()
    print("Took ", round(t2-t1, 3), "sec")


if __name__ == "__main__":
    main()



