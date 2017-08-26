# OpenCV+딥러닝을 적용해 연속 문자 인식

문자인식(Character Recognition)이란 시각 정보를 통하여 문자를 인식하고 의미를 이해하는 인간의 능력을 컴퓨터로 실현하려는 패턴인식(Pattern Recognition)의 한 분야로서, 광학 문자 인식(Optical Character Recognition), 우편물 자동 분류, 문서인식, 도면인식 등의 분야에서 부분적으로 실용화가 이루어지게 되었으며, 요즈음에는 인공지능(Artificial Intelligence)의 최신기법인 신경망(Neural Network)과 접목에 의해 문자인식 기술은 새로운 단계에 접어들게 되었습니다.


OpenCV와 딥러닝을 사용해 다음과 같은 이미지에 있는 숫자를 인식해보겠습니다. 여러개의 글자가 적혀 있으므로 이미지에서 문자가 어디 적혀 있는지 인식시켜야 합니다.

### 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593344612847

문자들이 어디 적혀 있는지 OpenCV를 이용하여 글자가 적힌 영역을 인식하는 방법을 알아 보겠습니다. 여기서 숫자 이미지 파일은 여러분들이 랜덤하게 선택해서 사용 하시면 됩니다. 여기서는 샘플로 numbers100.jpg를 사용 했습니다. 

openCV_digit_PI.ipynb 확인 하시면 됩니다.

```python

import sys, cv2 
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

# 이미지 읽어 들이기
im = cv2.imread('./photo/numbers100.jpg')

plt.imshow(im); plt.show()

# 그레이스케일로 변환하고 블러를 걸고 이진화하기
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# 윤곽 추출하기
# 두번째 매개변수를 cv2.RETR_LIST로 지정하면 모든 구간의 외곽을 검출합니다.
# 두번째 매개변수를 cv2.RETR_EXTERNAL로 지정하면 영역의 가장 외곽 부분만 검출합니다.
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

# 추출한 윤곽을 반복 처리하기
for cnt in contours:
  x, y, w, h = cv2.boundingRect(cnt)
  if h < 20: continue # 너무 작으면 건너뛰기
  red = (0, 0, 255)
  cv2.rectangle(im, (x, y), (x+w, y+h), red, 2)

cv2.imwrite('numbers-contour.png', im) 

image = mpimg.imread("numbers100-contour.png")
plt.imshow(image); plt.show()
```

## 1 문자 인식 데이터 만들기
각 문자 영역을 추출했으므로 각 자를 머신러닝으로 인식시켜 보겠습니다. MNIST의 손글씨 숫자 데이터를 Keras + TensorFlow를 이용하여  학습시켜 보겠습니다. 일단 MNIST의 손글씨 숫자 데이터를 학습하고 가중치 데이터를 저장합니다. 그런다음 저장한 가중치 데이터를 읽어드려 문자를 인식 시켜 보겠습니다. 

### 1.1 MNIST 데이터를 학습시키고, 가중치 데이터를 "mnist.hdf5"에 저장하는 프로그램입니다.

```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import h5py

# MNIST 데이터 읽어 들이기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# 데이터를 float32 자료형으로 변환하고 정규화하기

X_train = X_train.reshape(60000, 784).astype('float32')
X_test  =  X_test.reshape(10000, 784).astype('float32')

X_train /= 255; X_test /= 255


# 레이블 데이터를 0-9까지의 카테고리를 나타내는 배열고 변환하기

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test  = np_utils.to_categorical(Y_test , 10)



# 모델 구조 정의하기

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))


# 모델 구축하기


model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])



# 데이터 훈련하기

hdf5_file = "./hdf5/mnist-model.hdf5"

if os.path.exists(hdf5_file):
    model.load_weights(hdf5_file)

else:
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=600, epochs=10)
    model.save_weights(hdf5_file)



# 테스트 데이터로 평가하기

score = model.evaluate(X_test, Y_test, batch_size = 200, verbose=1)

print();
print("loss =", score[0], ", accuracy =", score[1], ", baseline error = %.2f%%" % (100-score[1]*100))

```

Result : loss = 0.0607551989534 , accuracy = 0.982700008154(98%) , baseline error = 1.73%

### 1.2 그림 이미지안에 숫자들을 1.1에서 학습된 MNIST(mnist.hdf5) 데이터로 인식 시키기

결론 부터 말씀 드리면, 당황스럽지만 다소 재미있는 결과가 나왔습니다.

숫자 폰트를 기반으로 만든 이미지 문자(약 71%)를 대부분 맞출 것이라 생각했는데 결과는 예상 밖으로 손으로 대충 적은 문자(약 98%)을 더 잘 예측했습니다. 깨끗하게 적은 숫자 데이터를 이 정도로 인식하지 못한다는 것은 굉장히 이상한 일입니다. 그리고 실행 결과를 보면 8 그리고 9라는 글씨를 가장 많이 제대로 인식하지 못했습니다. 아무래도 손글씨 8, 9과 폰트 8, 9이 잘 맞지 않는 모양입니다. 그럼 처음부터 폰트로 그린 이미지로 학습시켜보는 것은 어떨까요? 1.3에서 1.2의 약점을 보완해 보겠습니다.


```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

import h5py

import sys, cv2 

import numpy as np




#MLP 모델 구축
def build_model():

    # 모델 구조 정의하기
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    # 모델 구축하기
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

    return model


# MNIST 학습 데이터 읽어 들이기
mnist = build_model()


# 훈련된 데이터 사용하기
hdf5_file = "./hdf5/mnist-model.hdf5"
mnist.load_weights(hdf5_file)


# 이미지 읽어 들이기
im = cv2.imread('./photo/numbers100.jpg')

# 윤곽 추출하기
# Gray 스케일로 변환하기
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 2진화
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

cv2.imwrite("./photo/numbers100-th.png", thresh)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

# 추출한 좌표 정렬하기
rects = []
im_w = im.shape[1]


for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    # 너무 작으면 무시
    if w < 10 or h < 10: continue
    
    # 너무 크면 무시
    if w > im_w / 5: continue  

    # Y 좌표 맞추기
    y2 = round(y/10)*10 
    index = y2 * im_w + x

    rects.append((index, x, y, w, h))

# 정렬하기
rects = sorted(rects, key=lambda x:x[0]) 

X = []
for i, r in enumerate(rects):
    index, x, y, w, h = r

    num = gray[y:y+h, x:x+w] # 부분 이미지 추출하기
    num = 255 - num #반전하기

    # 정사각형 내부에 그림 옮기기
    ww = round((w if w > h else h) * 1.85)
    spc = np.zeros((ww, ww))

    wy = (ww-h)//2
    wx = (ww-w)//2

    spc[wy:wy+h, wx:wx+w] = num
    
    # MNIST 크기에 맞추기
    num = cv2.resize(spc, (28, 28))
    
    # 데이터 정규화
    num = num.reshape(28*28)
    num = num.astype("float32") / 255
    X.append(num)


# 예측하기
s="314159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230" +\
"664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593344612847"

answer = list(s)

correct = 0

nlist = mnist.predict(np.array(X))

for i, n in enumerate(nlist):
    ans = n.argmax()

    if ans == int(answer[i]):
        correct += 1

    else:
        print("[ng]", i, "번째", ans, " != ", answer[i], np.int32(n*100))

print("정답률 : ", correct / len(nlist))


----
실행 결과 :
[ng] 5 번째 4  !=  9 [ 0  0  0  0 68  0  0  3  0 28]
[ng] 11 번째 4  !=  8 [11  0  2  0 73  2  5  1  0  3]
[ng] 12 번째 4  !=  9 [ 0  0  0  0 68  0  0  3  0 27]
[ng] 14 번째 4  !=  9 [ 0  0  0  0 68  0  0  3  0 27]
[ng] 18 번째 4  !=  8 [11  0  2  0 73  2  5  1  0  3]
[ng] 26 번째 4  !=  8 [11  0  2  0 72  2  5  2  0  3]
[ng] 30 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 34 번째 4  !=  8 [11  0  2  0 72  2  5  2  0  3]
[ng] 35 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  3]
[ng] 38 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 42 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 44 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 45 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 50 번째 1  !=  0 [ 0 78  0  0  0  0  0 17  0  2]
[ng] 51 번째 0  !=  5 [62  0  0  0 23  0  0  8  0  5]
[ng] 52 번째 5  !=  8 [ 0  0  0  0  0 97  0  0  0  2]
[ng] 53 번째 4  !=  2 [11  0  2  0 72  2  5  1  0  2]
[ng] 54 번째 2  !=  0 [ 0  2 91  0  0  0  0  6  0  0]
[ng] 55 번째 0  !=  9 [62  0  0  0 23  0  0  8  0  5]
[ng] 56 번째 4  !=  7 [ 0  0  0  0 68  0  0  3  0 27]
[ng] 57 번째 7  !=  4 [ 0  0  2  0  0  0  0 96  0  0]
[ng] 58 번째 4  !=  9 [ 0  0  0  0 99  0  0  0  0  0]
[ng] 61 번째 4  !=  5 [ 0  0  0  0 99  0  0  0  0  0]
[ng] 62 번째 5  !=  9 [ 0  0  0  0  0 96  0  0  0  2]
[ng] 63 번째 4  !=  2 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 64 번째 2  !=  3 [ 0  2 90  0  0  0  0  6  0  0]
[ng] 65 번째 3  !=  0 [ 0  0  0 98  0  0  0  0  0  0]
[ng] 66 번째 0  !=  7 [61  0  0  0 24  0  0  8  0  5]
[ng] 67 번째 7  !=  8 [ 0  0  2  0  0  0  0 96  0  0]
[ng] 68 번째 4  !=  1 [11  0  2  0 72  2  6  1  0  2]
[ng] 74 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  2]
[ng] 78 번째 4  !=  8 [11  0  2  0 73  2  5  1  0  2]
[ng] 79 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 80 번째 4  !=  9 [ 0  0  0  0 68  0  0  3  0 27]
[ng] 81 번째 4  !=  8 [10  0  2  0 73  2  5  1  0  2]
[ng] 84 번째 4  !=  8 [10  0  2  0 73  2  5  1  0  2]
[ng] 88 번째 4  !=  8 [10  0  2  0 73  2  5  1  0  2]
[ng] 100 번째 4  !=  9 [ 0  0  0  0 68  0  0  3  0 27]
[ng] 101 번째 4  !=  8 [11  0  2  0 71  2  6  1  0  3]
[ng] 105 번째 4  !=  8 [11  0  2  0 71  2  6  1  0  3]
[ng] 107 번째 4  !=  8 [11  0  2  0 71  2  6  1  0  3]
[ng] 113 번째 4  !=  8 [11  0  2  0 71  2  6  1  0  3]
[ng] 122 번째 4  !=  9 [ 0  0  0  0 68  0  0  3  0 27]
[ng] 124 번째 4  !=  8 [11  0  2  0 71  2  5  1  0  3]
[ng] 129 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 134 번째 4  !=  8 [10  0  2  0 73  2  5  1  0  2]
[ng] 144 번째 4  !=  9 [ 0  0  0  0 68  0  0  3  0 28]
[ng] 147 번째 4  !=  8 [11  0  2  0 73  2  5  1  0  2]
[ng] 150 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  3]
[ng] 152 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  3]
[ng] 161 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  3]
[ng] 169 번째 4  !=  9 [ 0  0  0  0 69  0  0  3  0 27]
[ng] 171 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  3]
[ng] 180 번째 4  !=  9 [ 0  0  0  0 71  0  0  3  0 24]
[ng] 187 번째 4  !=  9 [ 0  0  0  0 70  0  0  3  0 26]
[ng] 189 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  2]
[ng] 190 번째 4  !=  9 [ 0  0  0  0 71  0  0  3  0 25]
[ng] 193 번째 4  !=  9 [ 0  0  0  0 70  0  0  3  0 26]
[ng] 197 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  2]
[ng] 199 번째 4  !=  9 [ 0  0  0  0 70  0  0  3  0 26]
[ng] 204 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  3]
[ng] 205 번째 4  !=  8 [11  0  2  0 73  2  5  1  0  3]
[ng] 208 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 214 번째 4  !=  9 [ 0  0  0  0 67  0  0  3  0 28]
[ng] 222 번째 4  !=  8 [11  0  2  0 72  2  5  1  0  3]

정답률 :  0.7111111111111111(71%)
```



### 1.3 다양한 숫자 폰트를 직접 만들어 학습 시키기

최근 OS에는 따로 설치하지 않아도 처음터 굉장히 많은 폰트가 있습니다. 디자인과 관련된 작업을 하는 분이라면 더 많은 폰트가 있을 텐데, OS에 설치돼 있는 모든 폰트를 사용해  숫자 폰트 이미지를 만든 다음 숫자 이미지를 학습시켜 정답률( > 71%)을 높여 보았습니다. 

아래 결과는 완전히 독립된 test set(number100.jpg) 자료로 계산한 결과입니다.

```python

[NG] 50 번째 1  !=  0 [  0 100   0   0   0   0   0   0   0   0]
[NG] 51 번째 0  !=  5 [100   0   0   0   0   0   0   0   0   0]
[NG] 52 번째 5  !=  8 [  0   0   0   0   0 100   0   0   0   0]
[NG] 53 번째 8  !=  2 [  0   0   0   0   0   0   0   0 100   0]
[NG] 54 번째 2  !=  0 [  0   0 100   0   0   0   0   0   0   0]
[NG] 55 번째 0  !=  9 [100   0   0   0   0   0   0   0   0   0]
[NG] 56 번째 9  !=  7 [  0   0   0   0   0   0   0   0   0 100]
[NG] 57 번째 7  !=  4 [  0   0   0   0   0   0   0 100   0   0]
[NG] 58 번째 4  !=  9 [  0   0   0   0 100   0   0   0   0   0]
[NG] 59 번째 9  !=  4 [  0   0   0   0   0   0   0   0   0 100]
[NG] 61 번째 4  !=  5 [  0   0   0   0 100   0   0   0   0   0]
[NG] 62 번째 5  !=  9 [  0   0   0   0   0 100   0   0   0   0]
[NG] 63 번째 9  !=  2 [  0   0   0   0   0   0   0   0   0 100]
[NG] 64 번째 2  !=  3 [  0   0 100   0   0   0   0   0   0   0]
[NG] 65 번째 3  !=  0 [  0   0   0 100   0   0   0   0   0   0]
[NG] 66 번째 0  !=  7 [100   0   0   0   0   0   0   0   0   0]
[NG] 67 번째 7  !=  8 [  0   0   0   0   0   0   0 100   0   0]
[NG] 68 번째 8  !=  1 [  0   0   0   0   0   0   0   0 100   0]

정답률 :  0.92(92%)
```

......
