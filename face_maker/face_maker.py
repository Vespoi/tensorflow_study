from PIL import Image
import os
import numpy as np
file_path = './../../face_img/img_align_celeba/img_align_celeba/'
file_list = os.listdir(file_path)

images = []

#crop -> 20,30 부터 160, 180 까지 잘라줌
for i in file_list[0:50000]:
    num_data = Image.open(file_path + i).crop( (20, 30, 160, 180) ).convert('L').resize( (64,64) )
    images.append( np.array(num_data) )

import matplotlib.pyplot as plt
plt.imshow(images[1])
plt.show()

images = np.array(images)
print(images.shape)

#흑백 이미지는 3차원이라 학습이 불가능
#1을 뒤에 추가해서 4차원으로 변경
images = np.divide(images, 255)
images.reshape( 50000, 64, 64, 1 )
print(images.shape)

#Discriminator
import tensorflow as tf

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding = 'same', input_shape=[64, 64, 1]),
    tf.keras.layers.LeakyReLU(alpha = 0.2), #활성함수
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding = 'same'),
    tf.keras.layers.LeakyReLU(alpha = 0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4 * 4 * 256, input_shape=(100,) ), 
  tf.keras.layers.Reshape((4, 4, 256)),
  tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])


generator.summary()

#간모델 -> 제네레이터와 디스크리미네이터도 하나의 레이어이다.
GAN = tf.keras.models.Sequential([
  generator,
  discriminator
])

#학습은 디스크리미네이터만 학습시킴
#오차는 바이너리 크로스엔트로피
discriminator.compile( optimizer = 'adam', loss = 'binary_crossentropy')

#앞에서 학습한거 GAN에서 학습 못하게 막음
discriminator.trainable = False

GAN.compile( optimizer= 'adam', loss = 'binary_crossentropy')

def predict_pic():
  #-1 부터 1사이를 균일하게 8개 들어있는 숫자를 100개
  predict = generator.predict(np.random.uniform(-1, 1, size = (8, 100) ) )

  print(predict.shape)

  for i in range(8):
    plt.subplot(2, 5, i+1)
    plt.imshow(predict[i].reshape(64,64), cmap = 'gray') #컬러라면 64, 64, 3
    plt.axis('off')
  plt.tight_layout()
  plt.show()

#-------------------
x_data = images

#discrimator 트레이닝
# 진짜사진 50장, 1로 마킹한 정답
# 가짜사진 50장, 0으로 마킹한 정답

# 아래 과정을 반복해서 5만장의 이미지를 모두 소비하자

batch = 128
epoch = 300
for j in range(epoch):

  print('now epoch {}'.format(j))

  for i in range(50000//batch):

    if i % 100 == 0:
      print('now batch is {}'.format(i))

    #진짜 사진 학습
    real_picture = x_data[ i * 128 : (i + 1) * 128 ]
    marking_one = np.ones(shape=(128, 1)) #1로 마킹

    discriminator.train_on_batch(real_picture, marking_one) #학습 
    loss1 = discriminator.train_on_batch(real_picture, marking_one) #오차 측정

    #가짜 사진 학습
    random_num = np.random.uniform(-1, 1, size = (128, 100) ) #가짜 사진만들때 사용할 숫자
    fake_picture = generator.predict(random_num) #가짜사진 만들기
    marking_zero = np.zeros(shape=(128, 1)) #0으로 마킹

    discriminator.train_on_batch(fake_picture, marking_zero) #학습
    loss2 = discriminator.train_on_batch(fake_picture, marking_zero) #오차 측정

    #편향을 줄이려면 위의 두 데이터를 섞어서 사용해도됨
    random_num = np.random.uniform(-1, 1, size = (128, 100) )
    marking_one = np.ones(shape=(128, 1))
    GAN.train_on_batch(random_num, marking_one )
    loss3 = GAN.train_on_batch(random_num, marking_one)

  print (f'이번 epoch의 최종 loss는 discriminator {loss1 + loss2} GAN {loss3}')

print('end of precedure')
predict_pic()