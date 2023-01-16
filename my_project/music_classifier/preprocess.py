import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

warnings.filterwarnings("ignore")

plt.rc('font', family = 'Gothic')

train = pd.read_csv('./../../../music/train.csv')
test = pd.read_csv('./../../../music/test.csv')

train.head()
train.info()
#데이터 크기 : 25383개
#Columns : 12개
#genre column 데이터타입 : object(범주형)
print(train)
print(train.isnull().sum()) #빈칸이 있는지 확인

# ------------------장르 라벨 ---------------------------
y_raw = train[['genre']]
print("학습시킬 데이터 라벨 : ", list(np.unique(y_raw)))
num_classes = len(np.unique(y_raw))

label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_raw)

print("숫자로 바뀐 장르 라벨 : ", np.unique(y_train))

print(y_train)

#y_train = keras.utils.to_categorical(y_train, num_classes)
#print(y_train)
#exit(1)

#X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.2)
# --------------------------------------------------------

# ------------ 나머지 데이터 딕셔너리화 -------------------
co_answer = train.pop('genre')
ds = tf.data.Dataset.from_tensor_slices( ( dict(train), y_train) )
ds_test = tf.data.Dataset.from_tensor_slices( (dict(test)))
print(ds) 

for i, l in ds.take(1):
    print(i)
    print(l)
#	danceability	energy	key	loudness	speechiness	acousticness	instrumentalness	liveness	valence	tempo	duration	genre

feature_columns = []
feature_columns.append(tf.feature_column.numeric_column('danceability'))
feature_columns.append(tf.feature_column.numeric_column('energy'))
feature_columns.append(tf.feature_column.numeric_column('key'))
feature_columns.append(tf.feature_column.numeric_column('loudness'))
feature_columns.append(tf.feature_column.numeric_column('speechiness'))
feature_columns.append(tf.feature_column.numeric_column('acousticness'))
feature_columns.append(tf.feature_column.numeric_column('instrumentalness'))
feature_columns.append(tf.feature_column.numeric_column('liveness'))
feature_columns.append(tf.feature_column.numeric_column('valence'))
feature_columns.append(tf.feature_column.numeric_column('tempo'))


print("피처 컬럼 출력")
#print(feature_columns)

#ds_batch = ds.batch(32)
#next(iter(ds_batch))[0]

#feature_layer = tf.keras.layers.DenseFeatures( tf.feature_column.numeric_column('tempo'))
#feature_layer(next(iter(ds_batch))[0])


"""
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation = 'softmax')
])
"""

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(num_classes, activation = 'softmax')
])

#model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

from tensorflow.keras.callbacks import TensorBoard # tensorboard --logdir logs
from tensorflow.keras.callbacks import EarlyStopping 
import time

tensroboard = TensorBoard( log_dir = 'logs/{}'.format( '첫모델' + str( int (time.time() ) )))

ds_batch = ds.batch(32)

es = EarlyStopping( monitor='val_accuracy', patience = 5, mode = 'max')


model.fit(ds_batch, shuffle=True, epochs = 150, callbacks = [tensroboard] )
model.save('./model4')

#model = tf.keras.models.load_model('./model1/')
submission = pd.read_csv("./../../../music/sample_submission.csv")

ds_test_batch = ds_test.batch(32)

model.evaluate(ds_test_batch)

predict = model.predict(ds_test_batch)
print(len(predict))

count = 0
index = []

for i in range(len(predict)):
    results = np.argsort(predict[count])[::-1]
    labels = label_encoder.inverse_transform(results)
    index.append(labels[0])

    count += 1


submission["genre"] = index
submission.to_csv("./submit.csv", index = False)



print("end of precedure")
