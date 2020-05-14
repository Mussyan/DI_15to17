# Affine Transfrom

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

# 이미지를 시계방향으로 theta 각도만큼 회전시키는 함수(음수를 넣으면 반시계 방향으로 회전)
def dbRotationV2(path, theta):
    deg = theta
    # 호도법 적용하여 라디안 형식으로 변환
    theta = math.radians(theta)
    origin = np.array(Image.open(path))
    img = origin.copy()
    
    # 세로(행 수), 가로(열 수), 채널 수
    h, w, c = img.shape[:3]
    
    # 내림 연산 - 중심 찾기
    # 이미지를 3배 확대하기 때문에 h*3
    center = math.floor((h*3)/2)-1
    
    # --- 이미지를 확대해 회전한 후, 원본 이미지의 크기로 돌려놓는다. --- #
    
    # 가로 세로 3배 크기의 빈 이미지 (84*84) 생성
    imgTemp = np.zeros((h*3, w*3, c), dtype=np.uint8)

    # 흰색으로 초기화 (세로, 가로, 채널 수 - 3)
    for i in range(0, h*3):
        for j in range(0, w*3):
            for k in range(0, c):
                imgTemp[i][j][k] = 255
    
    # 다시 화소를 낮춰서 저장할 이미지 (28*28) 생성
    img2 = np.zeros((h, w, c),dtype=np.uint8)
    
    # 흰색으로 초기화
    for i in range(0,h):
        for j in range(0,w):
            for k in range(0,c):
                img2[i][j][k] = 255

    # 행렬 연산을 활용하여 theta 각도만큼 시계방향으로 회전      
    for i in range(0, h*3):
        for j in range(0, w*3):
            check = 0
            # 흰색이 아닐 때
            if img[math.floor(i/3)][math.floor(j/3)][0] != 255:
                b = math.cos(theta)*j-center*math.cos(theta)-math.sin(theta)*i+center*math.sin(theta)+center
                a = math.sin(theta)*j-center*math.sin(theta)+math.cos(theta)*i-center*math.cos(theta)+center

                #계산결과 음수가 존재할 경우 화면 반대편의 좌표를 가리키게 된다
                if a>0 and b>0:
                    check = 1
                
                # 반올림
                a = round(a)
                
                # a가 이미지의 크기를 벗어나지 않게 한다
                if a > center*2:
                    a = (center*2)-1
                
                # 반올림
                b = round(b)
                
                # b가 이미지의 크기를 벗어나지 않게 한다
                if b > center*2:
                    b = (center*2)-1

                #a와 b가 전부 양수인 경우에만 회전, 음수가 존재할 경우 절삭
                if check == 1:
                    for k in range(0, c):
                        imgTemp[a][b][k] = img[math.floor(i/3)][math.floor(j/3)][0]

    # 반올림해서 생긴 비는 구간 보정
    for i in range(1, h*3-1):
        for j in range(1, w*3-1):
            if imgTemp[i-1][j][0] != 255 and imgTemp[i+1][j][0] != 255 and imgTemp[i][j-1][0] != 255 and imgTemp[i][j+1][0] != 255 and imgTemp[i][j][0] == 255:
                for k in range(0, c):
                    imgTemp[i][j][k] = (int(imgTemp[i-1][j][0])+int(imgTemp[i+1][j][0])+int(imgTemp[i][j-1][0])+int(imgTemp[i][j+1][0]))/4

    # 이미지 화질을 원본(28*28)로 낮추기(Average pooling 사용)
    for i in range(0, h):
        for j in range(0, w):
            temp = 0
            for k in range(0, 3):
                for l in range(0, 3):
                    temp += int(imgTemp[i*3+k][j*3+l][0])
            temp = math.floor(temp/9)
            
            for k in range(0, c):
                img2[i][j][k] = temp

    #이미지 저장(이름 레이블링 포함)
    deg = (int)((deg+9)/3)
    if(deg>3):
        deg = deg-1
    path2 = path[path.find(')'):]
    if(path.find('_')==-1):
        path1 = path[0:path.find(')')]
        pathSave = path1 + '_0_' + str(deg) + path2
    else:
        path1 = path[0:path.find(')')-2]
        pathSave = path1 + '_' + str(deg) + path2
    print(pathSave)
    imwrite(pathSave, img2)

#-9~9도 만큼 3도 간격으로 회전시키며 데이터 뻥튀기하는 함수
def runRotate(path):
    for i in range(-9, 10, 3):
        if(i!=0):
            dbRotationV2(path, i)
        
