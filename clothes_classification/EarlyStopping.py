import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#텐서 플로우에서 제공하는 옷 사진 튜플 받아오기
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

#데이터 출력해보기 -> list 안에 포함된 데이터는 모두 픽셀 단위 정보이다.
#print(trainX[0])

#데이터의 형태 출력
#(크기, 가로길이, 세로길이)
#print(trainX.shape)

#데이터의 라벨 출력
#print(trainY)

#    trainY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankelboot']

#plt.imshow( trainX[10] )
#plt.gray() -> 흑백으로
#plt.colorbar() -> 컬러 농도 표시 바
#plt.show() 

#선택사항
#이미지는 0 ~ 255 크기인데 학습시간 단축을 위하여 0 ~ 1 사이의 값으로 바꿔준다
#trainX = trainX / 255.0
#trainX = trainX / 255.0

#numpy를 이용하여 trainX의 모양을 바꾸자 3dim을 4dim으로 바꿔주자
trainX = trainX.reshape( ( trainX.shape[0], 28, 28, 1) )
testX = testX.reshape( ( testX.shape[0], 28, 28, 1) )

#학습----------------------------------------------------------

# [ 0.2, 0.4, 0.1, 0.1, ... ]

#tf.keras.layers.Flatten() = 2차원 list를 1차원으로 바꿔줌 (나열)
#tf.keras.layers.Conv2D( 몇개의 다른 feature, (kernel 가로세로 사이즈), padding = "종류", activation = "relu" )
#kernel 사이즈는 기본 (3, 3)에서 시작하여 점차 키워감
#padding -> 컨볼루션하면 작아지는 이미지 크기에 padding을 추가함(바깥쪽에 1px씩 더해주세요 같은 느낌 )
#왜 하필 relu? -> 이미지는 0부터 255 사이의 데이터이다. 그래서 음수가 나오지 않게 하기 위해서 relu (relu는 0보다 작을때 전부다 0이다)
#tf.keras.layers.MaxPooling2D( (크기, 크기), ) -> 크기를 압축하고, 중간으로 모은다.(부분에서 가장 큰 데이터를 기준으로 함)

#기본적으로 2차원 상침



#레이어를 추가해보거나 해서 학습하면서 시행착오를 거침

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32, (3, 3), padding="same", activation= "relu", input_shape=(28, 28, 1) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Conv2D( 64, (3, 3), padding="same", activation= "relu", input_shape=(28, 28, 1) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    #tf.keras.layers.Dense(128, input_shape=(28,28), activation = "relu"),
    tf.keras.layers.Flatten(), #출력은 1차원으로 할것이라서
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax") #sigmoid의 여러개 버전
])

#model outline 출력 (레이어별 특징을 표로 보여준다)
#반드시 첫번째 레이어에 input_shape(x, y) 넣어주자
#Param # -> 가중치와 편향의 총 갯수
#model.summary()

#exit()

#sparse_categorical_crossentropy vs categorical_crossentropy
#categorical_crossentropy는 one-hot incoding이 되어있는 데이터에만 사용가능
model.compile( loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'] )

from tensorflow.keras.callbacks import TensorBoard
import time
time.time()

tensorboard = TensorBoard( log_dir = 'logs/{}'.format( '첫모델' + str( int(time.time()) ) ) )

model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 3, callbacks = [tensorboard])

#epochs 마다 실제 데이터와의 차이를 평가함 -> overfitting을 줄이는데 도움이 된다.
#model.fit(trainX, trainY, validation_data=(testX, testY), epochs = 10)

#마지막 accuracy와 evaluate의  accuarcy는 차이가 있는데, 이것은 정형화된 데이터 학습의 문제이다.
score = model.evaluate( testX, testY )
print(score)