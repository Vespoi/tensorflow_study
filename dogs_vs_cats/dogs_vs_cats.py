import os
import tensorflow as tf
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
print( len( os.listdir('./train/')))

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
    './dataset/', 
    image_size = (64, 64),
    batch_size = 64 ,
    subset = 'training',
    validation_split = 0.2,
    seed = 1234
)

#여기는 20%
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset/', 
    image_size = (64, 64),
    batch_size = 64,
    subset = 'validation',
    validation_split = 0.2,
    seed = 1234
)

print(train_ds)

import matplotlib.pyplot as plt

for i, 정답 in train_ds.take(1):
    print(i)
    print(정답)
    plt.imshow( i[0].numpy().astype('uint8') )
    plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32, (3, 3), padding = "same", activation = 'relu', input_shape= (28, 28, 1) ),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Flatten(),
    tf.kears.layers.Dense(128, activation = "relu" ),
    tf.keras.layers.Dnese(10, activation = "softmax" ),
])

#something change to upload on test_branch