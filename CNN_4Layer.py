import tensorflow as tf
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
tf.set_random_seed(777)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from PIL import Image
from numpy import *




#해당 주석에서 말하는 단계는 카톡에 올려준 사진을 기준으로 한다!

#입력 이미지 shape(28x28)
img_rows, img_cols = 28,28

path1 = 'C:/Users/gang3/Desktop/JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format/training-images/'    #path of folder of images    
path2 = 'C:/Users/gang3/Desktop/JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format/training-images-resized/'  #path of folder to save images    

listing = os.listdir(path1)
num_samples=size(listing)
print('데이터 전처리 시작~')  

for file in listing:
    im = Image.open(path1 + '/' + file)  
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')                 
    gray.save(path2 +'/' +  file, "bmp")
    print('데이터 전처리중~ing'+file)
imlist = os.listdir(path2)

im1 = array(Image.open('C:/Users/gang3/Desktop/JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format/training-images' + '/'+ imlist[0])) # open one image to get size

m,n = im1.shape[0:2]#height,width
imnbr = len(imlist) 


immatrix = array([array(Image.open('C:/Users/gang3/Desktop/JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format/training-images-resized'+ '/' + im2)).flatten()


for im2 in imlist],'f')
          


  
#글자 레이블 세팅
label=np.ones((num_samples,),dtype = int)


#원본 데이터 레이블링
label[0:7996]=0#ㄱ
label[7997:12825]=1#ㄲ
label[12826:21145]=2#ㄴ
label[21146:21370]=3#ㄷ
label[21371:21545]=4#ㄸ
label[21546:21710]=5#ㄹ
label[21711:21804]=6#ㅁ
label[21805:21960]=7#ㅂ
label[21961:22104]=8#ㅅ
label[22105:22151]=9#ㅆ
label[22152:22407]=10#ㅇ
label[22408:22487]=11#ㅍ
label[22488:22544]=12#ㅋ
label[22545:22600]=13#ㅌ
label[22601:22706]=14#ㅎ
label[22707:26780]=15#ㅏ
label[26781:33868]=16#ㅓ
label[33869:33938]=17#ㅔ
label[33939:33987]=18#ㅖ
label[33988:38722]=19#ㅗ
label[38723:38895]=20#ㅘ
label[39986:38953]=21#ㅛ
label[38954:43355]=22#ㅜ
label[43356:43491]=23#ㅠ
label[43492:43657]=24#ㅡ
label[43658:43725]=25#ㅢ
label[43726:43774]=26#ㅣ


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img=immatrix[167].reshape(img_rows,img_cols)

print('데이터 작업 완료!')


#batch_size to train
batch_size = 32
# 10가지 중에서 무엇인지 one-hot encoding으로 출력
nb_classes = 27# number of epochs to train
nb_epoch = 10


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0],img_rows*img_cols)
X_test = X_test.reshape(X_test.shape[0],img_rows*img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#학습률
learning_rate = 0.001
#전체 학습 횟수
training_epochs =50
#학습 할 때 얼만큼의 데이터만큼 잘라서 학습할지 정한다.
batch_size = 100

#Dropout의 크기를 정하기 위한 변수
keep_prob = tf.placeholder(tf.float32)

print('학습 레이어 세팅\n');

# 비트맵 이미지의 Input Layer
X = tf.placeholder(tf.float32, [None, 784])
#비트맵 이미지를 재조정(인자 설명 : 몇개의 이미지 미정, 가로 28,세로28, 색깔 1개)
X_img = tf.reshape(X, [-1, 28, 28, 1])   


Y = tf.placeholder(tf.float32, [None, 27])

#1st Trial
#필터의 크기 : 3x3, 1=색깔, 32=필터 개수, stddeb=미분률?
#W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#
##1단계 작업 시작
##첫번째 Conv Layer
#L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
#L1 = tf.nn.relu(L1)
#
##첫번째 Max_pooling Layer-->비트맵 값중에서 가장 큰것을 색출-->이미지 크기 줄이고 
##Subsampling하는 효과!
##ksize=2x2
##strides=2x2이기 때문에 이미지의 크기가 반으로 줄어들게 된다.
#L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
#                    strides=[1, 2, 2, 1], padding='SAME')
##dropout 적용
#L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
##    Conv     -> (?, 28, 28, 32)
##    Pool     -> (?, 14, 14, 32)
##1단계 처리 완료 결과!(윗부분)
#
#
#
#
##2단계 시작->현재 이미지 상태 (n개, 14,14,32개 필터)
##필터 : 3x3, 32는 1단계에서의 필터 개수와 동일해야 한다. 64는 현재 정하는 필터의 개수
#W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#
#
##두번째 Conv Layer
#L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
#L2 = tf.nn.relu(L2)
##두번째 MaxPooling Layer, 필터:2x2, 간격이동:2칸씩==>이미지의 크기 2배 줄어든다.
#L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
#                    strides=[1, 2, 2, 1], padding='SAME')
#L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
##두번째 처리 결과
##    Conv      ->(?, 14, 14, 64)
##    Pool      ->(?, 7, 7, 64)
#
#
#
## L3 ImgIn shape=(?, 7, 7, 64)
##3단계 시작
##필터 : 3x3, 64는 2단계에서의 필터 개수와 동일해야 한다. 
##128은 현재 정하는 필터의 개수
#W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#
##3단계 Conv Layer
#L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
#L3 = tf.nn.relu(L3)
##3단게 MaxPooling Layer, 필터크기 :2x2, 간격이동 :2칸씩-->이미지의 크기 2배로 축소
#L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
#                    1, 2, 2, 1], padding='SAME')
#L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
#
#
#W3_3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
#
##3단계 Conv Layer
#L3_3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
#L3_3 = tf.nn.relu(L3)
##3단게 MaxPooling Layer, 필터크기 :2x2, 간격이동 :2칸씩-->이미지의 크기 2배로 축소
#L3_3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
#                    1, 2, 2, 1], padding='SAME')
#L3_3 = tf.nn.dropout(L3, keep_prob=keep_prob)
#
#
##현재 3단계 처리 완료된 상황의 픽셀을 벡터형으로 늘어 놓기(Fully Connected Layer 1)
#L3_flat = tf.reshape(L3, [-1, 256 * 4 * 2])
#
##입력 노드 개수 : 128*4*4, 출력 노드 개수 :625개
#W4 = tf.Variable(tf.random_normal([256 * 4 * 2, 625],stddev=0.01))
#                     
##bias도 출력노드 개수와 같이 설정
#b4 = tf.Variable(tf.random_normal([625]))
#L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
#L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
#L4_flat=tf.reshape(L4,[-1,625])
##Fully Connected Layer 1 완료
#
#
##Fully Connected Layer 2 시작
#W5 = tf.Variable(tf.random_normal([625, 124],stddev=0.01))
#                     
#b5 = tf.Variable(tf.random_normal([124]))#bias=출력 노드 개수
#L5=tf.nn.relu(tf.matmul(L4_flat,W5)+b5)
#L5 = tf.nn.dropout(L5, keep_prob=keep_prob)
#L5_flat=tf.reshape(L5,[-1,124])
#
#W7 = tf.Variable(tf.random_normal([124,62],stddev=0.01))
#                     
#b7 = tf.Variable(tf.random_normal([62]))#bias=출력 노드 개수
#L7=tf.nn.relu(tf.matmul(L5_flat,W7)+b7)
#
#
##Fully Connected Layer 3
#W6=tf.Variable(tf.random_normal([62,27],stddev=0.01))
#b6=tf.Variable(tf.random_normal([27]))
#logits = tf.matmul(L7, W6) + b6





#2nd Trial
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

#1단계 작업 시작
#첫번째 Conv Layer
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

#첫번째 Max_pooling Layer-->비트맵 값중에서 가장 큰것을 색출-->이미지 크기 줄이고 
#Subsampling하는 효과!
#ksize=2x2
#strides=2x2이기 때문에 이미지의 크기가 반으로 줄어들게 된다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
#dropout 적용
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
#1단계 처리 완료 결과!(윗부분)




#2단계 시작->현재 이미지 상태 (n개, 14,14,32개 필터)
#필터 : 3x3, 32는 1단계에서의 필터 개수와 동일해야 한다. 64는 현재 정하는 필터의 개수
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))


#두번째 Conv Layer
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
#두번째 MaxPooling Layer, 필터:2x2, 간격이동:2칸씩==>이미지의 크기 2배 줄어든다.
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
#두번째 처리 결과
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)



# L3 ImgIn shape=(?, 7, 7, 64)
#3단계 시작
#필터 : 3x3, 64는 2단계에서의 필터 개수와 동일해야 한다. 
#128은 현재 정하는 필터의 개수
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))

#3단계 Conv Layer
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
#3단게 MaxPooling Layer, 필터크기 :2x2, 간격이동 :2칸씩-->이미지의 크기 2배로 축소
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)





W3_3 = tf.Variable(tf.random_normal([3, 3,128,256], stddev=0.01))

#3단계 Conv Layer
L3_3 = tf.nn.conv2d(L3, W3_3, strides=[1, 1, 1, 1], padding='SAME')
L3_3 = tf.nn.relu(L3_3)
#3단게 MaxPooling Layer, 필터크기 :2x2, 간격이동 :2칸씩-->이미지의 크기 2배로 축소
L3_3 = tf.nn.max_pool(L3_3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3_3 = tf.nn.dropout(L3_3, keep_prob=keep_prob)




#현재 3단계 처리 완료된 상황의 픽셀을 벡터형으로 늘어 놓기(Fully Connected Layer 1)
L3_flat = tf.reshape(L3_3, [-1, 256 * 2 * 2])

#입력 노드 개수 : 128*4*4, 출력 노드 개수 :625개
W4 = tf.Variable(tf.random_normal([256 * 2 * 2, 625],stddev=0.01))
                     
#bias도 출력노드 개수와 같이 설정
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
L4_flat=tf.reshape(L4,[-1,625])
#Fully Connected Layer 1 완료


#Fully Connected Layer 2 시작
W5 = tf.Variable(tf.random_normal([625, 124],stddev=0.01))
                     
b5 = tf.Variable(tf.random_normal([124]))#bias=출력 노드 개수
L5=tf.nn.relu(tf.matmul(L4_flat,W5)+b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)
L5_flat=tf.reshape(L5,[-1,124])

W7 = tf.Variable(tf.random_normal([124,62],stddev=0.01))
                     
b7 = tf.Variable(tf.random_normal([62]))#bias=출력 노드 개수
L7=tf.nn.relu(tf.matmul(L5_flat,W7)+b7)


#Fully Connected Layer 3
W6=tf.Variable(tf.random_normal([62,27],stddev=0.01))
b6=tf.Variable(tf.random_normal([27]))
logits = tf.matmul(L7, W6) + b6



#비용 및 최적화 변수 선언 및 초기화
#보통 AdamOptimizer사용 많이 한다(GradientDescentOptimizer보다)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=Y,logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#세션 선언 및 실행하기(실행 전 초기화 필수!)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#모델 학습 시키기
#위에서 정한 Epoch값 만큼 전체 데이터 순환한다.



print('이미지 학습 시작')
for epoch in range(100):
    avg_cost = 0
    #total_batch = (전체 트레이닝 데이터 개수 / 미니 배치 크기)
    total_batch = int(X_train.shape[0]/batch_size)

    for i in range(total_batch):
        #트레이닝 데이터 설정
        batch_x=np.array([X_train[i].tolist()])
        batch_y=np.array([Y_train[i].tolist()])
        #값 설정 및 dropout 값 설정(0.7-->70%의 Weight만 사용한다는 뜻!)
        feed_dict = {X: batch_x, Y: batch_y, keep_prob: 0.5}
        
        #세션 Run!
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        
        
        #평균 비용
        avg_cost += c / total_batch
        
        #현재 학습 상황 보여주는 출력문들!
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('학습 완료!')


#얼마나 정확한지 정확도 분석하기
#logits의 결과와, Y값(0~9) 비교 
#여기에서는 keep_prob:1==>모든 weight를 사용한다.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('학습 정확도:', sess.run(accuracy, feed_dict={X: X_test, Y:Y_test, keep_prob: 1}))











##테스트용 데이터 중에서 하나 랜덤으로 선택한다.
#r = random.randint(0, len(Y_test) - 1)
##선택한 값 출력
#print("선택한 레이블 값: ", sess.run(tf.argmax(X_test[r:r + 1], 1)))
##학습 모델이 예측한 값 출력
##여기에서는 keep_prob:1==>모든 weight를 사용한다.
#print("모델이 예측한 값: ", sess.run(
#    tf.argmax(logits, 1), feed_dict={X: Y_test[r:r + 1], keep_prob: 1}))

    
#for i in range(10):
#    n=np.random.randint(647)
#    plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
#    plt.show()
#    print('정답 인덱스 : ',sess.run(tf.argmax(logits, 1), feed_dict={X: X_test[n:n + 1],Y:Y_test[n:n+1], keep_prob: 1}))
#    print('\n')
#    



#numbers1="[0]  [1]  [2]  [3]  [4]   [5]"
#char1=" ㄱ   ㄲ   ㄴ   ㄷ   ㄸ    ㄹ"
#numbers2="[6]  [7]  [8]  [9]  [10] [11]"
#char2=" ㅁ   ㅂ   ㅅ   ㅆ    ㅇ   ㅊ"
#numbers3="[12] [13] [14] [15] [16] [17]"
#char3=" ㅋ   ㅌ   ㅍ   ㅎ   ㅏ    ㅑ"
#numbers4="[18] [19] [20] [21] [22] [23]"
#char4=" ㅓ   ㅔ   ㅖ   ㅗ   ㅘ    ㅚ"
#numbers5="[24] [25] [26] [27] [28] [29]"
#char5=" ㅛ   ㅜ   ㅠ   ㅡ   ㅢ    ㅣ"
#
#
#for i in range(3):
#    n=np.random.randint(len(X_test))
#    plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
#    plt.show()
#    print('<<Labels List>>')
#    print(numbers1+'\n'+char1)
#    print(numbers2+'\n'+char2)
#    print(numbers3+'\n'+char3)
#    print(numbers4+'\n'+char4)
#    print(numbers5+'\n'+char5)
#    print('정답 인덱스 : ',sess.run(tf.argmax(logits, 1), feed_dict={X: X_test[n:n + 1],Y:Y_test[n:n+1], keep_prob: 1}))
#    print('\n')
    
