import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU를 쓰도록 강제
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    #scikit-learn 설치
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


pd.set_option('display.unicode.east_asian_width',True )
pd.set_option('display.max_columns', 20)
df = pd.read_csv('./crawling_data/naver_headline_news20220527.csv')
# print(df.head())
# df.info()

X = df['titles']
Y = df['category']

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

labeled_Y = encoder.transform(Y)
# print(labeled_Y)
label = encoder.classes_
# print(label)

# with open('./models/encoder.pickle', 'wb') as f:# wb 는 write binary
#     pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)

okt = Okt()
# okt_morph_X = okt.morphs(X[7], stem = True)
# print(okt_morph_X)

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
# print(X[:10])

stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)

for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
# 숫자를 바꿔줄 때 사용하는 것이 토크나이저
# print(X[:5])
# print(words)

with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_X = token.texts_to_sequences(X)
# wordsize = len(token.word_index) + 1
# print(tokened_X[:5])
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 17:
        tokened_X[i] = tokened_X[i][:17]

X_pad = pad_sequences(tokened_X, 17)
print((X_pad[:5]))

model = load_model('./models/news_category_classfication_model_0.6978114247322083.h5')
preds = model.predict(X_pad)
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts

print(df.head(30))
# hdf5 에러 발생 시 찾아가서 지우면 된다.

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = '0'
    else:
        df.loc[i, 'OX'] = 'X'
print(df.head(30))

print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))

for i in range(len(df)):
    if df['category'][i] != df['predict'][i]:
        print(df.iloc[i])