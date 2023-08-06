#!/usr/bin/env python
# coding: utf-8

# In[35]:



import os
import time
import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import math
from datetime import datetime
import numpy as np
#import winsound
import cv2 as cv
from tkinter import *
from PIL import Image, ImageTk

import tensorflow as tf
from PIL import Image, ImageOps
import imagecodecs
import cv2
import skimage
# for visualizations
import matplotlib.pyplot as plt

import numpy as np # for using np arrays
from numpy import asarray
import time

from skimage.io import imread, imshow
from skimage.transform import resize

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')
global path
global l
global Button_start,labelImg,labelImg1
def output(prediction,y):
    global labelImg1
    img1 = y.squeeze()
    img2 = prediction.squeeze()
    img2 = np.ma.masked_where(img2 == 0, img2)
    #plt.imshow(test_img)
    plt.figure(figsize=(10,10))
    plt.axis('Off')
    #test_mask = k[0]
    plt.imshow(img1 , cmap = 'gray' , interpolation = 'none')
    plt.imshow(img2 , interpolation = 'none', alpha = 0.5)
    #cv2.imwrite(r"C:/Users/Shankhadeep/OneDrive/Documents/model/hype.tif",preds_test_thresh[0])
    #k[0].save(r"C:/Users/Shankhadeep/OneDrive/Documents/model/hype.tif")
    plt.savefig("predict.png", bbox_inches='tight', pad_inches=0)
    time.sleep(2)
    tk.Label(root,text="Output Image",font=('calibre',12,'bold'),bg="#AFDCEC",fg='purple').place(x=470,y=300)
    img=cv2.imread("predict.png",0)
    img=cv2.resize(img,(250,250),interpolation = cv2.INTER_LINEAR)
    im = Image.fromarray(img)
    imgTk = ImageTk.PhotoImage(image=im)
    labelImg1 = tk.Label(root, image=imgTk)
    labelImg1.image = imgTk
    labelImg1.place(x=400, y=340)
    #Button_start.destroy()
    Button_start.configure(text="Reset!",command=lambda:reset(),font=('calibre',12,'bold'))
    tk.Label(root,text="Predicted",font=('calibre',12,'bold'),bg="#AFDCEC",fg='green').place(x=340,y=300)

def Conv2D_Block(input_tensor , n_filters):
  x = tf.keras.layers.Conv2D(filters = n_filters , kernel_size = (3 , 3) , kernel_initializer = 'he_normal' , padding = 'same')(input_tensor)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters = n_filters , kernel_size = (3 , 3) , kernel_initializer = 'he_normal' , padding = 'same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  return x

val=0.2
def U_Net(img_tensor , n_filters = 3):
  conv1 = Conv2D_Block(img_tensor , n_filters * 1)
  pool1 = tf.keras.layers.MaxPooling2D((2 , 2))(conv1)
  pool1 = tf.keras.layers.Dropout(val)(pool1)

  conv2 = Conv2D_Block(pool1 , n_filters * 2)
  pool2 = tf.keras.layers.MaxPooling2D((2 , 2))(conv2)
  pool2 = tf.keras.layers.Dropout(val)(pool2)

  conv3 = Conv2D_Block(pool2 , n_filters * 4)
  pool3 = tf.keras.layers.MaxPooling2D((2 , 2))(conv3)
  pool3 = tf.keras.layers.Dropout(val)(pool3)

  conv4 = Conv2D_Block(pool3 , n_filters * 8)
  pool4 = tf.keras.layers.MaxPooling2D((2 , 2))(conv4)
  pool4 = tf.keras.layers.Dropout(val)(pool4)

  conv5 = Conv2D_Block(pool4 , n_filters * 16)

  pool6 = tf.keras.layers.Conv2DTranspose(n_filters * 8 , (3 , 3) , (2, 2) , padding = 'same')(conv5)
  pool6 = tf.keras.layers.concatenate([pool6 , conv4])
  pool6 = tf.keras.layers.Dropout(val)(pool6)
  conv6 = Conv2D_Block(pool6 , n_filters * 8)

  pool7 = tf.keras.layers.Conv2DTranspose(n_filters * 4 , (3 , 3) , (2 , 2) , padding = 'same')(conv6)
  pool7 = tf.keras.layers.concatenate([pool7 , conv3])
  pool7 = tf.keras.layers.Dropout(0.01)(pool7)
  conv7 = Conv2D_Block(pool7 , n_filters * 4)

  pool8 = tf.keras.layers.Conv2DTranspose(n_filters * 2 , (3 , 3) , (2 , 2) , padding = 'same')(conv7)
  pool8 = tf.keras.layers.concatenate([pool8 , conv2])
  pool8 = tf.keras.layers.Dropout(val)(pool8)
  conv8 = Conv2D_Block(pool8 , n_filters * 2)

  pool9 = tf.keras.layers.Conv2DTranspose(n_filters * 1 , (3 , 3) , (2 , 2) , padding = 'same')(conv8)
  pool9 = tf.keras.layers.concatenate([pool9 , conv1])
  pool9 = tf.keras.layers.Dropout(val)(pool9)
  conv9 = Conv2D_Block(pool9 , n_filters * 1)

  output = tf.keras.layers.Conv2D(1 , (1 , 1) , activation = 'sigmoid')(conv9)

  u_net = tf.keras.Model(inputs = [img_tensor] , outputs = [output])

  return u_net



from keras import backend as K
def dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)
    return coef

def dice_loss(y_true, y_pred):
    loss = 1 - dice(y_true, y_pred)
    return loss


# In[5]:
def preprocessData(path):
    X = np.zeros((1, 128, 128, 1), dtype=np.float32)  

    img = tf.keras.preprocessing.image.load_img(path, grayscale=True)
    in_img = tf.keras.preprocessing.image.img_to_array(img)
    in_img = skimage.transform.resize(in_img , (128 , 128 , 1) , mode = 'constant' , preserve_range = True)
    X[0] = in_img / 255.0
    
    img_tensor = tf.keras.layers.Input((128 , 128 , 1) , name = 'img')
    model = U_Net(img_tensor)
    model.compile(optimizer = tf.keras.optimizers.Adam(),
            loss=dice_loss,
                  metrics=[dice])
    model.load_weights('unet.h5')
    #print("hi")
    #print(len(X))
    #plt.imshow(X[0])
    #fig, arr = plt.subplots(1,2, figsize=(15, 15))
    #arr[0].imshow(X[0])
    #arr[0].set_title('Processed Image')
    prediction= model.predict(np.expand_dims( X[0], 0))
    #print("hello")
    output(prediction,X[0])
    
def submit(): 
    global l
    global Button_start,labelImg
    #C:\Users\Shankhadeep\leena.png
    tk.Label(root,text="Input Image",font=('calibre',12,'bold'),bg="#AFDCEC",fg='purple').place(x=170,y=300)
   
    #path=(r"I:/Brachialplexus_Database-20230330T051557Z-001/Brachialplexus_Database/10_1.tif")
    path=l.get()
    #path=(r)
    '''
    img = Image.open(path)
    imgTk = img.resize((250, 250),Image.ANTIALIAS)
    imgTk = ImageTk.PhotoImage(imgTk)
    '''
    try:
        img=cv2.imread(path,0)
        img=cv2.resize(img,(250,250))#,interpolation = cv2.INTER_LINEAR)
        im = Image.fromarray(img)
        imgTk = ImageTk.PhotoImage(image=im)
        labelImg = tk.Label(root, image=imgTk)
        labelImg.image = imgTk
        labelImg.place(x=100, y=340)
        # Process data using apt helper function
        
        Button_start.configure(text="Predict!",command=lambda:preprocessData(path),font=('calibre',12,'bold'))
    
    except:
        reset()
# In[ ]:





# In[37]:
def reset():
    tk.Label(root,text="                           ",font=('calibre',12,'bold'),bg="#AFDCEC").place(x=170,y=300)
    tk.Label(root,text="                           ",font=('calibre',12,'bold'),bg="#AFDCEC").place(x=340,y=300)
    tk.Label(root,text="                           ",font=('calibre',12,'bold'),bg="#AFDCEC").place(x=470,y=300)
    try:
        labelImg1.destroy()
        labelImg.destroy()
        
    except:
        main()
    finally:
        main()
def main():
    
    global Button_start
    root.title("U-NET")
    root.geometry("700x600")
    #root.geometry("550x300+300+150")
    root.resizable(width=True, height=True)
    root.configure(background="#AFDCEC")
    global l
    
    #initialize
    #k=tk.StringVar()
    l=tk.StringVar()
    
    
    #Label & Entry
    label1=tk.Label(root,text="Brachial Plexus Segmentation",font=('calibre',12,'bold'),bg="#AFDCEC",fg='red').place(x=250,y=50)
    #label2=tk.Label(root,text="Input Time Limit (in seconds): ",font=('calibre',12,'bold'),bg="yellow").place(x=10,y=100)
    #box2=tk.Entry(root,textvariable=k,font=('calibre',10,'normal')).place(x=320,y=100)
    
    label3=tk.Label(root,text="Enter location of Image: ",font=('calibre',12,'bold'),bg="#AFDCEC").place(x=100,y=150)
    box3=tk.Entry(root,width=40,textvariable=l,font=('calibre',10,'normal')).place(x=300,y=150)
    
    
    #Button
    Button_start=tk.Button(root,text="Insert!",command=submit,font=('calibre',12,'bold'))
    Button_start.place(x=350,y=220)
    
    root.mainloop()

if __name__=="__main__":
    root=tk.Tk()
    main()
# In[ ]:




