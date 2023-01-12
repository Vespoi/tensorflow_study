"""
author : Kim min-gwan
date : 23.01.12
detail : 쇼핑몰 댓글을 이용한 악플 검사기 만들기
"""

import os
import pandas as pd
import numpy as np
import warnings
#경고문 삭제
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

#text = open("./../../naver_shopping.txt", "rt", encoding="utf-8").read()


#------------------데이터 입력 및 bag of words 만들기 ---------------------------------------------

raw = pd.read_table("./../../naver_shopping.txt", names = [ "rating", "review" ])
#print(raw)

#삼항연산자 같은 느낌
raw['label'] = np.where( raw['rating'] > 3, 1, 0)
#print(raw)

#데이터를 이쁘게 다듬자

#print(raw['review'].type())

#특수문자와 공백 제거
raw['review'] = raw['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '')
#print(raw)

#공백만 있는 행 제거
#print(raw.isnull().sum())

#중복 데이터 제거
raw.drop_duplicates( subset=['review'], inplace=True)
#print(raw)

#유니크한 문자모으기
#bag of words
unique_text = raw['review'].tolist()
unique_text = ''.join(unique_text)
unique_text = list(set(unique_text))
unique_text.sort()
#print(unique_text[0:100])


#------------------- 학습에 사용가능하게 전처리 ----------------------------------------
from tensorflow.keras.preprocessing.text import Tokenizer

#문자들을 정수로 바꾸자
# oov_toekn = '<OOV> -> train데이터를 정수화한다. 첨보는 단어 치환방법 <>로 구분하여 알아보기
tokenizer = Tokenizer(char_level = True, oov_token='<OOV>')

text_list = raw['review'].tolist()
tokenizer.fit_on_texts(text_list)

#tokenizer.word_index -> 딕셔너리로 만들어줌
word_index = tokenizer.word_index
#print(text_list[0:10])

#전체 단어에서 1회정도만 출현한 단어는 제거해주는게 좋음
#우리는 전부다 가져다 박음

train_seq = tokenizer.texts_to_sequences(text_list)
#print(train_seq[0:10])

Y = raw['label'].tolist()
#print(Y[0:10])

#모든 데이터 글자수 맞추기

raw['length'] = raw['review'].str.len()

#제일 긴 문장이 몇자인지 확인
#라벨이 포함하는 데이터의 수는 비슷해야좋음
#print(raw.head())
#print(raw.describe())

#최대 140자로 확인됨
#100~120자로 통일시키기

raw['length'][raw['length'] < 100].count()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#모든 길이 100자로 제한, 100자 안되면 0으로 체워넣음
X = pad_sequences(train_seq, maxlen = 100)
X = X.tolist()  #list로 안만들면 오류남 -> 텐서플로우에서 사용가능하게 타입 일치

#train/test/val 쪼개기
#test_size = 0.2 -> 20%
#random_state -> random seed
trainX, valX, trainY, valY = train_test_split(X, Y, test_size = 0.2, random_state=42 )


#---------------------------------- 학습 ----------------------
#print(len(trainX))
#print(len(valX))

import tensorflow as tf

#input_shape를 확인하기 위해 출력
#print (np.array(trainX).shape)

"""

#유니크한 문자가 3천개인 데이터는 원핫인코딩이 어렵다
model = tf.keras.Sequential([
    tf.keras.layers.Embedding( len(word_index) + 1, 16),
    tf.keras.layers.LSTM( 100, input_shape = (159540, 100) ),
    tf.keras.layers.Dense( 1, activation = 'sigmoid') #결과는 시그모이드
])

#시그모이드를 사용했으니 binary_corssentropy

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(trainX, trainY, batch_size = 64, epochs = 5)

model.save('./model1') #모델 save
#결과 : 정확도 92% 모델 완성

"""
#print(trainX[0:10])
#print("val data")
#print(valX[0:10])

#--------------------테스트 및 결론 -----------------------------------------------

#모델 불러오기
model = tf.keras.models.load_model('./model1/')
predict = model.predict(valX)

#숫자로 된 데이터를 텍스트로 변경
test_sentence = tokenizer.sequences_to_texts(valX)
#print(test_sentence)

test_list = []

#늘려준 크기 줄이고, 공백 줄여서 보기 좋게
for sample in test_sentence:
    temp = sample.replace('<OOV>', '')
    test_list.append(temp.replace('      ', ''))
    
#테스트 포인트
start = 10
end = 15

#조사해본 문자열
print("test")
print(test_list[start:end])

#실확률
print("predictions")
print(predict[start:end])

result = []

#50%를 넘어가면 정상, 아니면 악플

print("결과")
for i in range(start, end):
    if(predict[i] > 0.5):
        print(i, '정상')
    else:
        print(i, '악플')

        
#끝

