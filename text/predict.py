import os
import pandas as pd
import numpy as np
import tensorflow as tf

fp = pd.read_table("./input.txt", names = ["raw_input"])
print(fp)


fp['raw_input'] = fp['raw_input'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '')
print(fp)


unique_text = fp['raw_input'].tolist()
unique_text = ''.join(unique_text)
unique_text = list(set(unique_text))
unique_text.sort()

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(char_level = True, oov_token = '<OOV>')

text_list = fp['raw_input'].tolist()
tokenizer.fit_on_texts(text_list)

word_index = tokenizer.word_index

test_seq = tokenizer.texts_to_sequences(text_list)

print(test_seq)

word_token = tokenizer.sequences_to_texts(test_seq)

print(word_token)


#model = tf.keras.models.load_model('./model1/')
#predict = model.predict(fp['raw_input'])

#print(predict)



