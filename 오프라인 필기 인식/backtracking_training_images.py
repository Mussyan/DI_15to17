# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:57:49 2020

@author: gang3
"""
import numpy as np
import PIL
import matplotlib.pyplot as plt
import os

images_path='C:/Users/gang3/Desktop/JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format/training-images/'

global index
index=0
#비트멥 이미지 좌표 역추적 함수
def convert_coordinates(full_fname,index,fname):
   
    src=PIL.Image.open(full_fname).convert('1')
    src=src.convert('RGB')
    
    pixels=src.load()
    
    x=list()
    y=list()
    temp_x=list()
    temp_y=list()
    original_height=1000
    original_width=800
    
    #28,28 bitmap
    h,w=src.size 
    h_y=np.zeros_like(h)
    w_x=np.zeros_like(w)
    
    #resized
    ratio_h=original_height/h
    ratio_w=original_width/w
    
    ratio_x=np.zeros_like(round(ratio_w))
    ratio_y=np.zeros_like(round(ratio_h))
    
    for hh in range (h):
        for ww in range (w):
                if pixels[hh,ww]==(0,0,0):
                    #Calculate Ratio of pre screen
                    temp_h=round(hh*ratio_h,2)
                    temp_w=round(ww*ratio_w,2)
                    temp_x.append(temp_w)
                    temp_y.append(temp_h)
                    x.append(ww)
                    y.append(hh)
    
    
    #Transpose and flip array to make it correct way
    for t in range(len(temp_x)):
        ratio_x=np.append(ratio_x,temp_x[t])
        ratio_y=np.append(ratio_y,temp_y[t])
    
    for xx in range(len(x)):
        w_x=np.append(w_x,x[xx])
        h_y=np.append(h_y,y[xx])    
    
    #Change to 2d array(x,y)    
    ratio_x=ratio_x.reshape((-1,1))
    ratio_y=ratio_y.reshape((-1,1))
    resized_array=np.concatenate((ratio_x,ratio_y),axis=1)
    plt.scatter(ratio_x,ratio_y)
    plt.show()
    
    ##0,0 set left_top position
    #ax=plt.gca()                            # get the axis
    #ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    #ax.xaxis.tick_top()                     # and move the X-Axis      
    #ax.yaxis.set_ticks(np.arange(0, max(ratio_y), 50)) # set y-ticks
    #ax.yaxis.tick_left()       
    
    h_y=h_y.reshape((-1,1))
    w_x=w_x.reshape((-1,1))
    not_resized_array=np.concatenate((w_x,h_y),axis=1)
    plt.scatter(w_x,h_y)
    plt.show()
    
    
    ##0,0 set left_top position
    #ax=plt.gca()                            # get the axis
    #ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    #ax.xaxis.tick_top()                     # and move the X-Axis      
    #ax.yaxis.set_ticks(np.arange(0, max(h_y), 5)) # set y-ticks
    #ax.yaxis.tick_left()       
    #
    #plt.scatter(w_x,h_y)
    #plt.show()
    
    
    #Write down coordinates on txt files inside folder named backtrack_coordinates 
    f=open('C:/Users/gang3/Desktop/JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format/backtrack_coordinates/'+(fname)+'('+str(index)+')-resized.txt','w')
    f.write('X_coordinates &&  Y_coordinates')
    f.write('\n\n')
    f.write(str(resized_array))
    f.close()
    
    
    f2=open('C:/Users/gang3/Desktop/JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format/backtrack_coordinates/'+(fname)+'('+str(index)+')_origin.txt','w')
    f2.write('X_coordinates &&  Y_coordinates')
    f2.write('\n\n')
    f2.write(str(not_resized_array))
    f2.close()

    
    #Clear list!
    x.clear()
    y.clear()
    temp_x.clear()
    temp_y.clear()
    
for root, dirs, files in os.walk(images_path):
    for fname in files:
        full_fname = os.path.join(root, fname)
        fname=fname.split()
        fname=fname[0][0]
        convert_coordinates(full_fname,index,fname)
        index+=1
        print(fname+" 좌표 역추적 생성 완료")