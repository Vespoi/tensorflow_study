import tensorflow as tf
import os

model = tf.keras.models.load_model('./model1')



text = open('./../../pianoabc.txt', 'r').read()
#print(text)

#중복을 허용하지 않는 리스트 -> set
unique_text = list(set(text))
unique_text.sort()

#print(text)

#utilities
#utilitie 만들어두는게 국룰
text_to_num = {}
num_to_text = {}

for i, data in enumerate(unique_text):
    text_to_num[data] = i #글자를 숫자로 바꾸기
    num_to_text[i] = data


#print(text_to_num)

print(text_to_num['A']) # -> 1
#print(num_to_text[1]) # -> A

num_text = []
text_num = []

#숫자화 된 데이터 입력
for i in text:
    num_text.append( text_to_num[i] )


print( num_text )

import numpy as np

first_num = num_text[117 : 117 + 25]
first_num = tf.one_hot(first_num, 31)

#디멘션 크기 문제를 해결하기 위해 3차원으로 변경해준다.
first_num = tf.expand_dims(first_num, axis = 0)

#print(first_num)

music = []

#예측하는 부분
for i in range(200):

    predict = model.predict(first_num)
    predict = np.argmax(predict[0]) #가장큰 데이터를 뽑아온다.
    #print(predict)

    music.append(predict)

    #맨앞의 문자를 제거함
    next_num = first_num.numpy()[0][1:]
    #print(next_num)

    #one hot incoding
    one_hot_num = tf.one_hot(predict, 31)

    #두개의 리스트를 합쳐서 완전한 예측값을 생성함
    #원래 리스트에서 맨앞의 인덱스를 빼고, 맨뒤에 새로운 인덱스를 추가함
    first_num = np.vstack([ next_num, one_hot_num.numpy() ])
    first_num = tf.expand_dims(first_num, axis = 0)

print(music)

music_text = []

#문자로 바꿔서 음악 생성
for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))




