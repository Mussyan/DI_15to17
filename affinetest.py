import numpy as np
import cv2
from PIL import Image


def transformation():
     img=np.array(Image.open('DIGit/img/ㄱ(0).bmp'))
     h, w = img.shape[:2]
     
     M1=cv2.getRotationMatrix2D((w/2, h/2),340,1)
     
     img2=cv2.warpAffine(img,M1,(w,h), borderValue=(255,255,255))
     img2=cv2.resize(img2,(28,28))
     
     cv2.imshow('340-rotated',img2)
     
     
     #다른 이미지로 저장하기
     
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     
 
transformation()
