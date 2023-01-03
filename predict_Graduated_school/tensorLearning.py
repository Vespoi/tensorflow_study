import tensorflow as tf
import pandas as pd

#전처리 -----------------------------------------------------------------------

#pandas로 csv를 열겠음
#data라는 변수에 모두 담기(list 형태)
data = pd.read_csv('gpascore.csv')
print(data)

#받아온 data에서 빈칸이 있는 행을 모두 출력
#print(data.isnull().sum())

#data list에 포함된 데이터들 중에 공백이 포함된 행 모두 버리기
data = data.dropna()

#data list에 포함된 데이터들 중에 공백에 지정된 값 집어넣기
#data.fillna(집어넣을값)


#print(data['gpa'].min()) -> gpa 열에 가장 작은값
#print(data['gpa'].max()) -> gpa 열에 가장 큰값
#print(data['gpa'].count()) -> gpa 열의 멤버 수
#print(data['gpa'].values) -> gpa 열의 멤버


y_data = data['admit'].values
x_data = []

#pandas iter as row -> iterrows() 한행씩 찍어줌
#pandas iter as col -> itercols() 한열씩 찍어줌

for i, rows in data.iterrows():
    #print(rows['gre']) #해당 열에 gre를 출력
    #x_data = []
    x_data.append([ rows['gre'], rows['gpa'], rows['rank'] ])

#데이터 구조
#ex) x_data [ [380, 3.21,3] [600, 3.67, 3] [] [] ]
#ex) y_data [ [0] [1] [1] ... ]

#print(x_data)

#여기까지만 실행
#exit()

#학습 ------------------------------------------------------------------------

# model making
# tf.keras.models.Sequential([레이어 상세])
# tf.keras.layers.Dense(노드 갯수, activation = '활성함수 종류')

#활성함수 -> ex) sigmoid, tanh, relu, softmax

model = tf.keras.models.Sequential([  
    tf.keras.layers.Dense(64, activation = 'relu'), # 중간 레이어 1번 64개의 노드
    tf.keras.layers.Dense(128, activation = 'relu'), # 중간 레이어 2번 128개 노드
    tf.keras.layers.Dense(1, activation  = 'sigmoid'),  # 마지막 레이어는 노드 한개
])

#실수로 출력됨 (정수 X)

#model compile
#model.compile(optimizer = 'w값 변화시킬 방법', loss = '손실함수 종류', metrics = ['결과 평가 요소'])
#optimizer -> ex) adam
model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

import numpy as np

#model learning
#model.fir(numpy array, numpy array, epochs = '학습시킬 횟수')
model.fit( np.array(x_data), np.array(y_data), epochs = 1000)



#예측------------------------------------------------------------

#앞에서 학습시킨 model로 예측하기
#model.predict( [ [], [] ] )
predict = model.predict( [ [750, 3.70, 3], [400, 2.2, 1] ] )
print(predict)




