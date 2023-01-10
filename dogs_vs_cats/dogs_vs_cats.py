import os
import tensorflow as tf
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
print( len( os.listdir('./../../train/')))

#데이터 옮기기 (cat vs dog)
'''
for i in os.listdir('./train/'):
    if 'cat' in i:
        shutil.copyfile( './train/' + i , './dataset/cat/' + i )
    if 'dog' in i:
        shutil.copyfile( './train/' + i , './dataset/dog/' + i )
'''

# Image Aumentation
# -> overfitting 줄이기 위해 이미지 변화주기 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,  #회전
    zoom_range = 0.15,   #확대
    width_shift_range = 0.2,  #이동
    height_shift_range = 0.2,
    shear_range = 0.15,  #굴절
    horiznotal_flip = True,  #가로반전
    fill_mode = "nearest"
)

generator_tarin = generator.flow_from_directory(
    './../../dataset',
    class_mode = 'binary', # 두개면 binary, 몇개더면 categorical
    shuffle = True,
    seed = 123,  #random seed
    color_mode = 'rgb', #  'gray'
    batch_size = 64,
    target_size = (64, 64)
)

생성기2 = ImageDataGenerator( rescale = 1./255 )

검증용 = 생성기2.flow_from_directory(
    './../../dataset',
    class_mode = 'binary',
    shuffle = True,
    seed = 123,  #random seed
    color_mode = 'rgb', #  'gray'
    batch_size = 64,
    target_size = (64, 64)
)

#---------------------------------------------------------------------------

#batch size -> epoch에 한번에 모든 데이터를 넣는게 아니고, batch size 만큼만 넣어줌
# train data = ( (xxxxxx (실제 데이터) ), (0 or 1 (정답) ) )
# seed -> random한 정도

# validation_split = 0.2 -> 80% 만큼 쪼개서 사용함
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './../../dataset/', 
    image_size = (64, 64),
    batch_size = 64 ,
    subset = 'training',
    validation_split = 0.2,
    seed = 1234
)

#여기는 20%
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './../../dataset/', 
    image_size = (64, 64),
    batch_size = 64,
    subset = 'validation',
    validation_split = 0.2,
    seed = 1234
)

print(train_ds)


# 모든 데이터를 255로 나눠서 0 ~ 1 사이로 압축하기
# 학습 속도를 빠르게 하기 위해서
def 전처리함수(i, 정답):
    i = tf.cast( i / 255.0, tf.float32 )
    return i, 정답

train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)

for i, 정답 in train_ds.take(1):
    print(i)
    print(정답)


#학습------------------------------------------------------

import matplotlib.pyplot as plt

for i, 정답 in train_ds.take(1):
    print(i)
    print(정답)
    plt.imshow( i[0].numpy().astype('uint8') )
    plt.show()

#input_shape -> RGB라서 64 64 3
#마지막 dense는 sigmoid -> binary_crossentropy는 sigmoid를 필요로 함
#마지막 dense는 개인지 고양이인지만 확인하면됨. 따라서 1개의 레이어에 sigmoid로 확률 출력
#Dropout(0.2) -> 현재 레이어에서 20%를 제거하여 정형화된 데이터 생성을 막음 -> overfitting을 막는다.

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'), # 이미지 뒤집기 (50% 확률)
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), # 이미지 돌리기 (50% 확률)
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1), # 이미지 확대 (50% 확률)

    tf.keras.layers.Conv2D( 32, (3, 3), padding = "same", activation = 'relu', input_shape= (64, 64, 3) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Conv2D( 64, (3, 3), padding = "same", activation = 'relu', input_shape= (64, 64, 3) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D( 128, (3, 3), padding = "same", activation = 'relu', input_shape= (64, 64, 3) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu" ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = "sigmoid" ),
])

model.summary()

model.compile( loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'] )
model.fit(train_ds, validation_data = val_ds, epochs = 5 )

model.evaluate(val_ds)


#저장하고 다시하기 --------------------------------------------------------

#checkpoint 저장 방법
#filepath -> 저장위치 ( {epoch} 마다 저장해줌, 안하면 덮어쓰기로 저장됨)
#save_weights_only = True -> 오로지 Weight만 저장
#save_freq = 'epoch' -> epoch마다 저장

"""

콜백함수 = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'checkPoint/mnist{epoch}',
    #monitor = 'val_acc' -> 정확도를 측정
    #mode = 'max' -> 가장 높은 정화도를 가진 weith를 저장
    save_weights_only = True,
    save_freq = 'epoch'
)
"""

#model 저장
#model.save('model_save/model1')

#불러온모델 = tf.keras.models.load_model('model_save.model1')
#불러온모델.summary()
#불러온모델.evaluate(val_ds)

#하이퍼 파라미터 튜닝?
#데이터의 양을 늘리거나 퀄리티를 높히면 정확도가 올라감

"""
model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32, (3, 3), padding = "same", activation = 'relu', input_shape= (64, 64, 3) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Conv2D( 64, (3, 3), padding = "same", activation = 'relu', input_shape= (64, 64, 3) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D( 128, (3, 3), padding = "same", activation = 'relu', input_shape= (64, 64, 3) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu" ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = "sigmoid" ),
])

model2.summary()

model2.compile( loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'] )

model2.load_weights('checkpoint/mnist')

model2.evaluate(val_ds)

"""
#model2.fit(train_ds, validation_data = val_ds, epochs = 5 )

