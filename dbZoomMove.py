#Zoom out and Move
import dbAffine as af
import numpy as np
import cv2
import math
from PIL import Image
import os

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 

        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
            return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False

#이미지 90프로 비율로 축소하는 함수
def zoomOut(path):
    
    origin = np.array(Image.open(path))
    img = origin.copy()
    
    h, w, c = img.shape[:3]
    
    img2 = np.zeros((h, w, c),dtype=np.uint8)
    
    # 흰색으로 초기화
    for i in range(0,h):
        for j in range(0,w):
            for k in range(0,c):
                img2[i][j][k] = 255

    #이미지 90프로 비율로 축소하여 저장
    for i in range(0, h):
        for j in range(0, w):
            for k in range(0, c):
                img2[math.floor(i*0.9)][math.floor(j*0.9)][k] = img[i][j][k]

    #이동 및 회전까지 실시하며 저장
    moveOn(path)

#이미지를 이동시키는 함수
def moveOn(path):
    
    origin = np.array(Image.open(path))
    img = [0,0,0,0]
    img[0] = origin.copy()

    h, w, c = img[0].shape[:3]

    #각각 오른쪽, 아래쪽, 대각 이동
    for i in range(1, 4):
        img[i] = np.zeros((h, w, c),dtype=np.uint8)
    
    # 흰색으로 초기화
    for i in range(0,h):
        for j in range(0,w):
            for k in range(0,c):
                for l in range(1,4):
                    img[l][i][j][k] = 255

    #각 이미지 이동하여 저장
    for i in range(0, h-3):
        for j in range(0, w-3):
            for k in range(0, c):
                if(img[0][i][j][k]!=255):
                    img[1][i+3][j][k] = img[0][i][j][k]
                    img[2][i][j+3][k] = img[0][i][j][k]
                    img[3][i+3][j+3][k] = img[0][i][j][k]

    #레이블링 포함 이미지 저장 및 회전 호출
    path1 = path[0:path.find(')')]
    path2 = path[path.find(')'):]
    pathSave = [0,0,0,0]
    for i in range(0,4):
        pathSave[i] = path1 + '_'+str(i+1)+'_0' + path2
        print(pathSave[i])
        imwrite(pathSave[i], img[i])
        af.runRotate(pathSave[i])
