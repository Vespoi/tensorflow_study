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


#something change to upload on test_branch