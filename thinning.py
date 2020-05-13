_#필요한 라이브러리 임포트 및 초기 설정
import cv2
import numpy as np
from PIL import Image
 
#세선화 알고리즘을 구현한 함수
#해당 이미지의 255 비트 개수를 기반으로
#글자의 굵기 판단
#현재 함수의 인자는 없다
#함수 내부에서 조건에 맞게끔
#인자의 값을 설정한다 


def Thinned_Image():
    
    #이미지 읽어오기
    src=np.array(Image.open('img/ㅜ(35).bmp'))
     
    #검은색 픽셀 개수 저장 변수 선언 및 초기화
    black=0
    height,weight,channel=src.shape[:3]
    
    #검은색 픽셀 개수 세기
    for i in range(0,height):
        for j in range(0,weight):
            for k in range(0,channel):
                if src[i][j][k] !=255:
                    #특정 검은색 픽셀 이상의 값 개수만 추출 
                    if src[i][j][k]>=110:
                        black+=1
    
    print('Black pixels : {0}'.format(black))
    
    #Done!
    #글씨 굵기에 따른 여러 경우
    if black<=100:
        degree=40
        k1=2
        k2=2
    
    elif black<=110:
        degree=40
        k1=3
        k2=3
    
    elif(black<=145):
        #임계값 설정
        degree=40
        #커널 사이즈 지정 
        k1=2
        k2=2
        
    elif black<155:
        #임계값 설정
        degree=10
        #커널 사이즈 지정
        k1=3
        k2=3
        
    elif black==150:
        degree=10
        k1=2
        k2=2
    
        
    elif black==180:
        degree=40
        k1=3
        k2=3
        
    elif black<=200:
        #임계값 설정
        degree=40
        #커널 사이즈 지정
        k1=2
        k2=2
         
        
    else:
        #임계값 설정
        degree=10
        #커널 사이즈 지정
        k1=2
        k2=2
         
   
    
    #이미지 반전
    src=cv2.bitwise_not(src)
    temp,A=cv2.threshold(src,degree,255,cv2.THRESH_BINARY)
     
    #변환 결과 저장할 공간 확보
    skel=np.zeros(src.shape,np.uint8)
    shape2=cv2.MORPH_RECT
    
    #비트 수 카운트 변수 선언 및 초기화
    count=0
    B=cv2.getStructuringElement(shape=shape2,ksize=(k1,k2))
    done=True
    
    
    while done:
        
        #변환한 비트 수 카운트 
        count+=1
         
        #erode->걸부분 깎아내기
        erode=cv2.erode(A,B)
        opening=cv2.morphologyEx(erode,cv2.MORPH_OPEN,B)
        tmp=cv2.subtract(erode,opening)
        skel=cv2.bitwise_or(skel,tmp)
        A=erode.copy()
        
        #28x28==784 모든 픽셀의 처리가 끝나게 되면 
        #boolean설정 변경하여 루프 탈출
        if count==784:
            done=False
        
        else:
            continue
    
    #이미지 재반전
    skel=cv2.bitwise_not(skel)
     
    #image 리사이즈( cv2 보간법 적용)
    skel=cv2.resize(skel,dsize=(28,28),interpolation=cv2.INTER_AREA)
    
    #변환된 이미지 출력
    cv2.imshow('skel_dst',skel)
    
    #변환된 이미지 다른이름으로 저장하기
    cv2.imwrite('testimage.bmp',skel)
    cv2.waitKey()
    
    #destroyAllWindows() 필요없습니다. 
    
#함수 호출
Thinned_Image()
 
 


