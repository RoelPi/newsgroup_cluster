# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:05:49 2017

@author: roel
"""
import pandas as p # My homie
import numpy as np

from sklearn.datasets import fetch_20newsgroups # dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # Vectorize terms

from sklearn.naive_bayes import MultinomialNB # For supervised learning
from gensim import corpora, models # For LDA
from sklearn.cluster import KMeans # For KMeans

from nltk.stem import PorterStemmer # Preprocess - For stemming
from stemming.porter2 import stem # Preprocess - Another stemming package
from nltk.tokenize import RegexpTokenizer # Preprocess
from stop_words import get_stop_words # Preprocess
import string # Preprocess
import re # Preprocess

import sklearn.metrics as skm # Evaluate unsupervised clustering

newsgroups_train = fetch_20newsgroups(subset='train',
                                     remove=('headers', 'footers', 'quotes'))
# a bunch = a simple holder object with fields that can be both accessed as 
# python dict keys or object attributes for convenience
# data = the tweet
# filenames = where the tweet is stored
# target = classification
# target_names = topics (name of the classification)
classes = np.array(newsgroups_train.target_names)
df = newsgroups_train.data

# STEMing
# Thanks at https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def preprocess_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

# Count occurances. Careful, it does not account for the fact that the
# documents are longer.
cv_transform = CountVectorizer(analyzer=preprocess_words, 
                               stop_words = 'english',
                               ngram_range = (1,3),
                               min_df=5)
dfo = cv_transform.fit_transform(newsgroups_train.data)

# In the CountVectorizer there is now a dictionary.
# dfv.vocabulary_.get('and')

# It is of course better to get densities, so that it accounts for the fact
# haven't got the same length. So we device occurences by total doc word count.
tf_transform = TfidfTransformer(use_idf=False).fit(dfo)
dff = tf_transform.transform(dfo)

# There is another refinement: adjusting frequency for unique occurence.
tfidf_transform = TfidfTransformer(use_idf=True)
dffi = tfidf_transform.fit_transform(dfo)

###########################################################
# Naive Bayes classification (Supervised) #################
###########################################################
dfModelNb = MultinomialNB().fit(dffi, newsgroups_train.target)

# Prediction of random stuff
dfNew = ['Church and prayer.',
         'Jesus is king.',
         'I drive a Harley.',
         'Wayne Gretzky likes ice and pucks.',
         'Give me my rifle.',
         'Allah loves Muslims.',
         'My car needs reparing.'
         ]
dfNew_o = cv_transform.transform(dfNew)
dfNew_fi = tfidf_transform.transform(dfNew_o)

dfNewP = classes[dfModelNb.predict(dfNew_fi)]
dfNewPred = p.DataFrame({'Text':dfNew,'Class':dfNewP})
del dfNewP

###########################################################
# K-means clustering (Unsupervised) #######################
###########################################################
clustModel = KMeans(n_clusters = 20, init = 'k-means++', n_init = 3, 
               max_iter = 100)

dfClustK = clustModel.fit_predict(dffi)
dfPredKMeans = p.DataFrame({'Text':newsgroups_train.data,'Class':classes[newsgroups_train.target],
                     'KMeans':dfClustK})
    
# Evaluation of the clusters, compared to their ground truth label.
dfPredEvalKMeans = skm.homogeneity_completeness_v_measure(dfPredKMeans.Class,dfPredKMeans.KMeans)

###########################################################
# Latent Dirichlet Allocation #############################
###########################################################
def prep(df):
    # Lowercase
    df = [i.lower() for i in df]
    
    # Remove punctuation
    df = [re.sub(r'[{}]'.format(string.punctuation),'',i) for i in df]
    
    # Remove numbers
    df = [re.sub(r'\d+','',i) for i in df]
    
    # Remove nextlines
    df = [i.strip() for i in df]
    df = [re.sub('\n|\r','',i) for i in df]
    
    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    df = [tokenizer.tokenize(i) for i in df]
    
    # Remove stop words
    en_stop = get_stop_words('en')
    df = [[i for i in tokens if not i in en_stop] for tokens in df]
    
    # STEMing
    df = [[stem(token) for token in tokens] for tokens in df]
    
    return df

def do_tdm(df):
    # Create Document Term Matrix (DTM)
    dictionary = corpora.Dictionary(df)
    corpus = [dictionary.doc2bow(text) for text in df]
    return corpus, dictionary

def do_lda(corpus, dictionary):
    dfLDA = models.LdaModel(corpus, num_topics=20, id2word=dictionary)
    return dfLDA

corpus, dictionary = do_tdm(prep(df))
ldaModel = do_lda(corpus, dictionary)

def get_cluster(df):
    dfPrep = prep(df)
    dfBow = [dictionary.doc2bow(dfPrepLine) for dfPrepLine in dfPrep]
    dfPreds = [ldaModel[i] for i in dfBow]
    dfPred = []
    for i in range(0,len(dfPreds)):
        prob = []
        cluster = []
        for j in dfPreds[i]:
            prob.append(j[1])
            cluster.append(j[0])
        dfPred.append(cluster[prob.index(max(prob))])
    return dfPred

dfClustLDA = get_cluster(df)

dfPredLDA = p.DataFrame({'Text':df,'Class':classes[newsgroups_train.target],
                     'LDA':dfClustLDA})
    
# Evaluation of the clusters, compared to their ground truth label.
dfPredEvalLDA = skm.homogeneity_completeness_v_measure(dfPredLDA.Class,dfPredLDA.LDA)