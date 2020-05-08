# On palmetto cluster - ncpus=10:ngpus=2:gpu_model=v100:mem=320gb,walltime=24:00:00
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from collections import Counter
from nltk import ngrams

import seaborn as sns
import string
import re

from sklearn.naive_bayes import MultinomialNB
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re

    

def main():
    
    
    file_path="Amazon_review.csv"
    data = pd.read_csv(file_path)
    
    
    data.columns = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer','date', 'dateAdded', 'dateSeen',
       'didPurchase', 'doRecommend', 'id','numHelpful', 'rating', 'sourceURLs','text', 'title', 'userCity',
       'userProvince', 'username']
    
    
    
    ratings = data.rating.value_counts()
    good=ratings[5.0]+ratings[4.0]
    not_good=ratings[3.0]+ratings[2.0]+ratings[1.0]
    total=ratings[5.0]+ratings[4.0]+ratings[3.0]+ratings[2.0]+ratings[1.0]
    #NPS Score count
    NPS_score = round ((100*(good-not_good)/total),2)
    print (" NPS score of Amazon is : "  + str(NPS_score))
    
    new_reviews=fake_review(data)
    data_int = data[['categories','rating' , 'text' , 'title' , 'username']]
    # new_reviews after fake reviews are deleted, data is orginal data set without removing fake reviews.
    #sentiment = new_reviews[new_reviews["rating"].notnull()]
    sentiment = data[data["rating"].notnull()]
    sentiment.head()


    #Assign sentiment based on review ratings
    sentiment["sentiment"] = sentiment["rating"]>=4
    sentiment["sentiment"] = sentiment["sentiment"].replace([True , False] , ["positive" , "negative"])
    
    # Data cleaning
    cleanup_re = re.compile('[^a-z]+')
    sentiment["Summary_Clean"] = sentiment["text"].apply(clean)
    naive_bayes(sentiment)
    other_algo(sentiment)
    lstm(sentiment)

    
   
   
def fake_review(data):    
    
    
    name_count= data.name.value_counts() 
    name_counts=name_count.keys()
    
    new_reviews=pd.DataFrame()
    
    #loop for each product 
    for k in range(len(name_counts)):
        #create temp data for each product count number of duplicate users in each
        temp=data[data["name"]==name_counts[k]]
        rating_perperson=temp.username.value_counts()
        #allow 4 reviews per user if user have given more than 4 reviews for same product means duplicate user
        bulk = rating_perperson[rating_perperson >= 4]
        user_names=bulk.keys()
        if bulk.empty:
            new_reviews=new_reviews.append(temp)
        else:
            for j in range(len(user_names)):
                r1=temp[temp["username"]!=user_names[j]]
                temp=r1
            new_reviews=new_reviews.append(temp)
                      
    
    
    print("Number of reviews before removing fake reviews:",data.shape[0])
    print("Number of reviews after removing fake reviews:",new_reviews.shape[0])
    fake_review = data.shape[0]-new_reviews.shape[0]
    print ("Fake reviews : ",fake_review)
  
    return new_reviews
    
    
    
def word_features(words, n=2):
    words_ngram = ngrams(words, n)
    features = {}
    for word in words_ngram:
        features [word] = True
    return features

def clean(sentence):
    cleanup_re = re.compile('[^a-z]+')
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    #sentence = " ".join(nltk.word_tokenize(sentence))
    return sentence



def naive_bayes(sentiment):
    
    # The cleaned data is split into train and test set
    split = sentiment[["Summary_Clean" , "sentiment"]]
    train=split.sample(frac=0.8,random_state=200)
    test=split.drop(train.index)
    train["words"] = train["Summary_Clean"].str.lower().str.split()
    test["words"] = test["Summary_Clean"].str.lower().str.split()
    train.index = range(train.shape[0])
    test.index = range(test.shape[0])
    prediction =  {} 

    train_naive = []
    test_naive = []
    
    # Labeling each word in the train and test dataset
    for i in range(train.shape[0]):
        train_naive = train_naive +[[word_features(train["words"][i]) , train["sentiment"][i]]]
    for i in range(test.shape[0]):
        test_naive = test_naive +[[word_features(test["words"][i]) , test["sentiment"][i]]]
    # Fit Naive Bayes model
    classifier = NaiveBayesClassifier.train(train_naive)
    print("NLTK Naive bayes Accuracy : {}".format(nltk.classify.util.accuracy(classifier , test_naive)))
    classifier.show_most_informative_features(10)




def other_algo(sentiment):
    stopwords = set(STOPWORDS)
    stopwords.remove("not")
    prediction={}
    split = sentiment[["Summary_Clean" , "sentiment"]]
    train=split.sample(frac=0.8,random_state=200)
    test=split.drop(train.index)
    # Generating tokens and calculating the tf-idf weights of each token
    count_vect = CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))
    tfidf_transformer = TfidfTransformer()

    X_train_counts = count_vect.fit_transform(train["Summary_Clean"])        
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


    X_new_counts = count_vect.transform(test["Summary_Clean"])
    X_test_tfidf = tfidf_transformer.transform(X_new_counts)
    # Fit multinomial Naive Bayes model
    model1 = MultinomialNB().fit(X_train_tfidf , train["sentiment"])
    prediction['Multinomial'] = model1.predict_proba(X_test_tfidf)[:,1]
    print("Multinomial Accuracy : {}".format(model1.score(X_test_tfidf , test["sentiment"])))

    # Fit Bernoulli Naive Bayes model
    model2 = BernoulliNB().fit(X_train_tfidf,train["sentiment"])
    prediction['Bernoulli'] = model2.predict_proba(X_test_tfidf)[:,1]
    print("Bernoulli Accuracy : {}".format(model2.score(X_test_tfidf , test["sentiment"])))

    # Fit logistic regression model
    logreg = linear_model.LogisticRegression(solver='lbfgs' , C=1000)
    logistic = logreg.fit(X_train_tfidf, train["sentiment"])
    prediction['LogisticRegression'] = logreg.predict_proba(X_test_tfidf)[:,1]
    print("Logistic Regression Accuracy : {}".format(logreg.score(X_test_tfidf , test["sentiment"])))


# we have run LSTM using palmetto
def lstm(sentiment):

	max_fatures = 30000
	# Genrate tokens
	tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
	tokenizer.fit_on_texts(sentiment['Summary_Clean'].values)

	# Initialize initial weights
	X1 = tokenizer.texts_to_sequences(sentiment['Summary_Clean'].values)
	X1 = pad_sequences(X1)

	Y1 = pd.get_dummies(sentiment['rating']).values
	X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
	print(X1_train.shape,Y1_train.shape)
	print(X1_test.shape,Y1_test.shape)

	embed_dim = 150
	lstm_out = 200
	# Building the neural network
	model = Sequential()
	model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1]))
	model.add(LSTM(lstm_out))
	model.add(Dense(5,activation='softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	print(model.summary())
	batch_size = 512
	# Training LSTM model
	model.fit(X1_train, Y1_train, epochs = 50, batch_size=batch_size, validation_split=0.3 , verbose = 2)
	# Calculating accuracy
	score,acc = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)
	print("score: %.2f" % (score))
	print("acc: %.2f" % (acc))
    
   
if __name__=="__main__":
    main()
  