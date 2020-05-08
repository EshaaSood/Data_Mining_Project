from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from collections import Counter
import nltk
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer
import re 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

sw = nltk.corpus.stopwords.words('english')
pstemmer = PorterStemmer()



def sentiment_score(x):
    score = SentimentIntensityAnalyzer().polarity_scores(x)
    score_count=score['compound']
    return score_count

# Word normalization
def stemmer (sent): 
    temp1 ="".join(x for x in sent if x not in string.punctuation)
    temp2 = re.split('\W+',temp1.lower())
    temp3 = [pstemmer.stem(x) for x in temp2 if x not in sw]
    return temp3

def convert(x):   
    if x==False:
        return 0
    else :
        return 1 
    
def main():
    
    #read dataset
    file_path="Amazon_review.csv"
    data = pd.read_csv(file_path)
    
    #removing null values
    data = data.dropna(subset=['categories','reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username'])
    data_interest1 = data[['categories','reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.doRecommend']]
    
    #feature extraction using stemmer 
    vs = TfidfVectorizer(analyzer=stemmer)
    features=vs.fit_transform(data_interest1['reviews.text'])
        
    # Feature 1 : Calculate Sentiment compound value for each of the reviews
    for  index, row in data_interest1.iterrows():
         data_interest1.set_value(index,'sentiment',sentiment_score(row['reviews.text']))

    # Feature 2 : Length of each of the review
    data_interest1['length'] = data_interest1['reviews.text'].apply(lambda x : len(re.split('\W+',x)))
    for  index, row in data_interest1.iterrows():
         data_interest1.set_value(index,'recommend',convert(row['reviews.doRecommend']))

    #reset index for data split
    new_sentiment = data_interest1.sentiment.reset_index()['sentiment']
    new_length = data_interest1.length.reset_index()['length']
    x_features = pd.concat([new_sentiment,new_length,pd.DataFrame(features.toarray(),columns=vs.vocabulary_)],axis=1)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_features,data_interest1.recommend,test_size=0.2)
    
    # Train model using random forest classifier
    rf = RandomForestClassifier(n_jobs=-1,n_estimators=40,max_depth=100)
    rfmodel=rf.fit(x_train,y_train)
    
    #predicted y-value from x-test value
    y_predict = rfmodel.predict(x_test)
    
    # Calculating precision, recall and accuracy
    precision, recall, fscore , support = score(y_test,y_predict,average='binary')
    accuracy=(y_predict==y_test).sum()/len(y_test)
    
    #final result 
    print('Precision: {} | Accuracy {} |Recall :{}  '.format(precision,accuracy,recall))

    
    
if __name__=="__main__":
    main()
  