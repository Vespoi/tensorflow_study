import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers as tfl
import os
import cv2
from sklearn import preprocessing
from pathlib import Path
from PIL import Image
pd.set_option('max_rows', 99999)
pd.set_option('max_colwidth', 400)
pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


#----------------------------------데이터 전처리 ------------------------------------------
def extract_label(base):
    #base -> 폴더 경로
    path = [] #파일 경로
    label = [] # 0L, 0R, 1L, 1R, 2L, 2R, 3L, 3R, 4L, 4R, 5L, 5R,
    #라벨과 파일 경로 입력
    for filename in os.listdir(base):
        label.append(filename.split('.')[0][-2:]) #맨끝에 붙어있는 라벨을 가져옴
        path.append(base + filename) #파일 위치와 파일 이름

    return path, label

#파일 경로
train_base  = "./../../../fingers/train/"
test_base = "./../../../fingers/test/"

#파일 경로와 라벨 받아오기
train_set_path, train_set_label = extract_label(train_base)
test_set_path, test_set_label = extract_label(test_base)

print("Number of training set examples : ", len(train_set_path))
print("Number of test set examples: ", len(test_set_path))

# 첫번째 이미지 미리보기-------------------------------
index = 0
image = cv2.imread(train_set_path[index])

plt.imshow(image)
plt.title(train_set_label[index], fontsize = 20)
plt.show()
#----------------------------------------------------

#데이터를 학습 가능한 형태로 분리시켜 저장
def feature_data_split(path):
    feature_set = []
    for p in path:
        image = cv2.imread(p)
        feature_set.append(image)
    return feature_set

#데이터 분리
X_train = feature_data_split(train_set_path)
X_test = feature_data_split(test_set_path)

X_train = np.array(X_train)
X_test = np.array(X_test)

print(X_train.shape)
print(X_test.shape)

#--------------- 라벨 처리  ------------------------------

print("학습시킬 데이터 라벨 :", list(np.unique(train_set_label)))
print("테스트할 데이터 라벨 :", list(np.unique(test_set_label)))
num_classes = len(np.unique(train_set_label))

#라벨을 숫자로 바꾸기 ex) 0L = 1, 0R = 2
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(train_set_label)
y_test = label_encoder.fit_transform(test_set_label)

print("숫자로 바뀐 학습   라벨 : ", np.unique(y_train))
print("숫자로 바뀐 테스트 라벨 : ", np.unique(y_test))

#원 핫 인코딩
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(y_test.shape)

#----------------학습 ------------------------------------
#컨볼루션 -> 맥스풀 -> 컨볼루션 -> 맥스풀 -> 플래튼 -> 덴스
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape = (128, 128, 3)),
    tf.keras.layers.MaxPool2D( (2, 2) ),
    tf.keras.layers.Conv2D(64, (3, 3), padding = "same", activation='relu', input_shape = (128, 128, 3)),
    tf.keras.layers.MaxPool2D( (2, 2) ),
    tf.keras.layers.Conv2D(128, (3, 3), padding = "same", activation='relu', input_shape = (128, 128, 3)),
    tf.keras.layers.MaxPool2D( (2, 2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.summary()

model.compile(loss = "categorical_crossentropy", optimizer= 'adam', metrics= ['accuracy'])

#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train).batch(128))
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test).batch(128))


#모델 세이브

from tensorflow.keras.callbacks import TensorBoard
import time

tensorboard = TensorBoard( log_dir = 'logs/{}'.format( '첫모델' + str( int( time.time()) ) ) )

model.fit(X_train, y_train, batch_size= 128, epochs=5, validation_data=(X_test, y_test), callbacks = [tensorboard])

model.save('./model1')

score = model.evaluate( X_test, y_test)
print("테스트 결과 :", score)


