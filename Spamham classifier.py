import nltk
import numpy as np
import pandas as pd
import re

messages=pd.read_csv("C:\\Users\\Anuj Kumar\\Desktop\\data science\\data set\Project\\SMSSpamCollection",sep='\t',names=['Labels','Messages'])

from nltk.corpus import  stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Messages'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000)
X=cv.fit_transform(corpus).toarray()
#one hot encoding to LABLES
Y=pd.get_dummies(messages['Labels'])
Y=Y.iloc[:,1].values


#Naive Baise Classifier

# SPLITTING DATA INTO TRAIN AND TEST

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.20,random_state=0)

#TRAINING MODEL ON TRAIN DATA
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(train_x,train_y)

#PREDICTION ON TESTING DATA

pred=model.predict(test_x)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix as cm
cof=cm(test_y,pred)
cof

#CHECKING ACCURACY
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(test_y,pred)
accuracy
#Accuracy of Model is 98.116 %