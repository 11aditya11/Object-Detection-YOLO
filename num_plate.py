import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from PIL import Image
from alpr import recognize_license_plate
from keras.models import load_model

def crop_num_plate(cropped_vehicle):
    #loading model
    model = load_model('numberplate.h5')
    WIDTH = 224
    HEIGHT = 224
    CHANNEL = 3

    #object detection cropped image
    #path = "test1.jpg"
    img = np.array(cropped_vehicle)
    img = cv2.resize(img / 255.0, dsize=(WIDTH, HEIGHT))

    y_hat = model.predict(img.reshape(1, WIDTH, HEIGHT, 3)).reshape(-1) * WIDTH

    xt, yt = y_hat[0], y_hat[1]
    xb, yb = y_hat[2], y_hat[3]
    
    #cropping number plate

    #loading cropped image
    crop_img = cropped_vehicle

    x_r = crop_img.size[0]/225
    y_r = crop_img.size[1]/225

    #extending cropped region by 10%

    new_x1 = xt*x_r - 0.13*crop_img.size[0]
    new_y1 = yt*y_r - 0.1*crop_img.size[0]
    new_x2 = xb*x_r + 0.14*crop_img.size[0]
    new_y2 = yb*y_r + 0.1*crop_img.size[0]
    
    crop_num_plate = crop_img.crop((new_x1, new_y1, new_x2, new_y2))
    label = 'numberplate'
    a,b,c,d, text = recognize_license_plate(np.array(crop_num_plate))
    a = new_x1 + a
    b = new_y1 + b
    c = new_x1 + c
    d = new_y1 + d
    crop_num_plate.save(label + ".jpg")
    print("number plate image saved") 
    return (a,b,c,d,text)

