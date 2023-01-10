import os

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
    #num_to_text[i] = data

#print(text_to_num)

print(text_to_num['A']) # -> 1
#print(num_to_text[1]) # -> A

num_text = []

#숫자화 된 데이터 입력
for i in text:
    num_text.append( text_to_num[i] )

#print( num_text )


X = []
Y = []

for i in range( 0, len(num_text) - 25 ):
    X.append( num_text[ i : i + 25 ] )
    Y.append(num_text[ i + 25 ])

#trainX = [ [25개], [25개] ]

print(X[0:5])
print(Y[0:5])

import numpy as np
print( np.array(X).shape)


import tensorflow as tf


#one - hot incoding
# ex) 4개 숫자중에 2번이면 2번째 자리에 1을 올림 ( [ 0, 0, 1, 0 ])

trainX = tf.one_hot(X, 31)
trainY = tf.one_hot(Y, 31)
print( trainX[ 0 : 2 ] )


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM( 100, input_shape = (25, 31) ), #LSTM 안에 활성함수가 많이 있음
    tf.keras.layers.Dense( 31, activation = 'softmax' )
])

#categorial_crossentropy -> softmax
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#verbose = 2 -> 출력을 많이 하지마세요 라는 의미
#epoch가 많을때 출력이 많으면 다운되는 현상이 발생함
model.fit(trainX, trainY, batch_size = 64, epochs= 60, verbose = 2)

model.save('./model1')
