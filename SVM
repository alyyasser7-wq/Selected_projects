import pandas as pd
import csv , cv2
import numpy as np
import tensorflow, keras
from keras.layers import Input, Conv2D, Activation, MaxPool2D,  Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import drive
drive.mount("/content/drive",force_remount=True)
%cd '/content/drive/MyDrive/selected'
dru_train = pd.read_csv('/content/drive/MyDrive/selected/drugsComTrain_raw.tsv', sep = "\t")
dru_test = pd.read_csv('/content/drive/MyDrive/selected/drugsComTest_raw.tsv', sep = "\t")
#type(dru_train)
len(dru_train)
print('Training length: ',dru_train.shape)
len(dru_test)
print('Testing length:',dru_test.shape)
dru = pd.concat([dru_train,dru_test])
dru.shape
dru.columns = ['Id','drugName','condition','review','rating','date','usefulCount']  
dru['date'] = pd.to_datetime(dru['date'])   
dru['date'].head()    
dru2 = dru[['Id','review','rating']].copy() 
dru2.head()
dru2.isnull().any().any()
dru.info()   
import nltk
nltk.download(['punkt','stopwords'])
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
dru2.head()
dru2['cleanReview'] = dru2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))  
dru2.head()
!pip install vaderSentiment  
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
dru2['vaderReviewScore'] = dru2['cleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
dru2.head()
positive_num = len(dru2[dru2['vaderReviewScore'] >=0.05])
neutral_num = len(dru2[(dru2['vaderReviewScore'] >-0.05) & (dru2['vaderReviewScore']<0.05)])
negative_num = len(dru2[dru2['vaderReviewScore']<=-0.05])
positive_num,neutral_num,negative_numdru2['vaderSentiment']= dru2['vaderReviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0) )
dru2['vaderSentiment'].value_counts()
Total_vaderSentiment = positive_num + neutral_num + negative_num
Total_vaderSentiment
dru2.loc[dru2['vaderReviewScore'] >=0.05,"vaderSentimentLabel"] ="positive"
dru2.loc[(dru2['vaderReviewScore'] >-0.05) & (dru2['vaderReviewScore']<0.05),"vaderSentimentLabel"]= "neutral"
dru2.loc[dru2['vaderReviewScore']<=-0.05,"vaderSentimentLabel"] = "negative"
positive_rating = len(dru2[dru2['rating'] >=7.0])
neutral_rating = len(dru2[(dru2['rating'] >=4) & (dru2['rating']<7)])
negative_rating = len(dru2[dru2['rating']<=3])
positive_rating,neutral_rating,negative_rating
Total_rating = positive_rating+neutral_rating+negative_rating
Total_rating
dru2['ratingSentiment']= dru2['rating'].map(lambda x:int(2) if x>=7 else int(1) if x<=3 else int(0) )
dru2['ratingSentiment'].value_counts()
dru2.loc[dru2['rating'] >=7.0,"ratingSentimentLabel"] ="positive"
dru2.loc[(dru2['rating'] >=4.0) & (dru2['rating']<7.0),"ratingSentimentLabel"]= "neutral"
dru2.loc[dru2['rating']<=3.0,"ratingSentimentLabel"] = "negative"
dru2.head()
dru2 = dru2[['Id','review','cleanReview','rating','ratingSentiment','ratingSentimentLabel','vaderReviewScore','vaderSentiment','vaderSentimentLabel']]
dru2.head()
dru2.to_csv('processed.csv')  
dru2.head(50)
import os
os.stat('processed.csv').st_size
dru2.info()
dru2.to_csv('processed.csv.gz',compression='gzip')
os.stat('processed.csv.gz').st_size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
dru2 = pd.read_csv('/content/drive/MyDrive/selected/processed.csv')
dru2.shape
dru2.head()
dru2.columns
tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
features = tfidf.fit_transform(dru2.cleanReview)
labels   = dru2.vaderSentiment
features.shape
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20)
normalize = Normalizer()
X_train = normalize.fit_transform(X_train)
X_test = normalize.transform(X_test)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred= model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
