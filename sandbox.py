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
from sklearn.decomposition import NMF # For Non-Negative Matrix Factorization
from sklearn.decomposition import LatentDirichletAllocation # For LDA
from sklearn.cluster import KMeans # For KMeans

from nltk.stem import PorterStemmer # Preprocess - For stemming

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

###########################################################
# Document Preprocessing ##################################
###########################################################

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
                               min_df=3)
dfo = cv_transform.fit_transform(newsgroups_train.data)

# In the CountVectorizer there is now a dictionary.
# dfv.vocabulary_.get('and')

# For NMF we need to get densities, so that it accounts for the fact that docs
# haven't got the same length. So we divide occurences by total doc word count.
tf_transform = TfidfTransformer(use_idf=False).fit(dfo)
dff = tf_transform.transform(dfo)

# For NMF there is another refinement: adjusting frequency for unique occurence.
tfidf_transform = TfidfTransformer(use_idf=True)
dffi = tfidf_transform.fit_transform(dfo)

###########################################################
# Handy functions #########################################
###########################################################

def find_cluster(a):
    return p.DataFrame(a).idxmax(axis=1)

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
# Latent Dirichlet Allocation (Unsupervised) ##############
###########################################################

# This is class for an object that contains information about an lda_model that has been run.
# The info is the topic_prior (alpha), the word_prior (eta), the model object, the predictions object and the NMI.
class lda_model_sweep:
    def __init__(self,lda_model,predictions,NMI,topic_prior,word_prior):
        self.topic_prior = topic_prior
        self.word_prior = word_prior
        self.lda_model = lda_model
        self.predictions = predictions
        self.NMI = NMI
    
    def get_topic_prior(self):
        return self.topic_prior
    
    def get_word_prior(self):
        return self.word_prior
    
    def get_model(self):
        return self.lda_model
    
    def get_predictions(self):
        return self.predictions
    
    def get_NMI(self):
        return self.NMI

# This is a function to run an LDA algorithm on a bow with alpha and eta.
# It returns an lda_model_sweep object.
def do_lda(dfo_train,dfo_test,topic_prior,word_prior):
    ldaModel = LatentDirichletAllocation(n_components=20, 
                                max_iter=5,
                                doc_topic_prior = topic_prior,
                                topic_word_prior = word_prior,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    dfClustLDA = find_cluster(ldaModel.fit_transform(dfo))
    dfPredLDA = p.DataFrame({'Text':newsgroups_train.data,
                  'Class':classes[newsgroups_train.target],
                  'LDA':dfClustLDA})
    dfPredEvalLDA = skm.normalized_mutual_info_score(dfPredLDA.Class,dfPredLDA.LDA)
    
    sweep_lda = lda_model_sweep(ldaModel,dfPredLDA,dfPredEvalLDA,
                         topic_prior,word_prior)
    return sweep_lda

# This is a function to sweep over the hyperparameters alpha and eta of an LDA.
# It returns a list of lda_model_sweep objects.
def do_hp_sweep_lda(dfo_train,dfo_test,topic_prior_list,word_prior_list):
    hp_sweep_list = [do_lda(dfo,topic_prior,word_prior) 
                    for topic_prior in topic_prior_list
                    for word_prior in word_prior_list]
    return hp_sweep_list

lda_sweeps = do_hp_sweep_lda(dfo,[0.0,0.25,],[0.75,1.0])
lda_overview = [[[i.get_model()],[i.get_word_prior()],[i.get_topic_prior()],[i.get_NMI()]] for i in lda_sweeps]
winning_model = lda_overview[
        [i[3] for i in lda_overview].index(max([i[3] for i in lda_overview])) # Get index of winning model
        ][0]

###########################################################
# Non-Negative Matrix Factorization (Unsupervised) ########
###########################################################

nmfModel = NMF(n_components=20, 
          random_state=1,
          alpha=.1,
          l1_ratio=.5)

dfClustNMF = find_cluster(nmfModel.fit_transform(dffi))
dfPredNMF = p.DataFrame({'Text':newsgroups_train.data,
                   'Class':classes[newsgroups_train.target],
                   'NMF':dfClustNMF})
dfPredEvalNMF = skm.normalized_mutual_info_score(dfPredNMF.Class,dfPredNMF.NMF)

###########################################################
# K-means clustering (Unsupervised) #######################
###########################################################
clustModel = KMeans(n_clusters = 20, init = 'k-means++', n_init = 3, 
               max_iter = 100)

dfClustK = clustModel.fit_predict(dffi)
dfPredKMeans = p.DataFrame({'Text':newsgroups_train.data,
                    'Class':classes[newsgroups_train.target],
                    'KMeans':dfClustK})
    
# Evaluation of the clusters, compared to their ground truth label.
dfPredEvalKMeans = skm.normalized_mutual_info_score(dfPredKMeans.Class,dfPredKMeans.KMeans)


    