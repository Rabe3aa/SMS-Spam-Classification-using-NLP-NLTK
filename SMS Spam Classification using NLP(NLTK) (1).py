#!/usr/bin/env python
# coding: utf-8

# In[441]:


import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from nltk import WordNetLemmatizer


# In[442]:


df = pd.read_csv('D:/Self studying/NLP/LinkedIn/Essential Training NLP/archive/SMSSpamCollection.tsv', header=None, sep='\t')


# In[443]:


df.head()


# In[444]:


df.columns = ['label', 'text']


# In[445]:


df.head()


# ## Remove Punctuations

# In[446]:


def remove_punctuations(text):
    text = text.lower()
    temp = ''.join([i for i in text if i not in string.punctuation])
    return temp


# In[447]:


df['text_punc'] = df['text'].apply(lambda x: remove_punctuations(x))


# In[448]:


df.head()


# ## Tokenize

# In[449]:


def tokenize(text):
    temp = re.split('\W+', text)
    return temp


# In[450]:


df['text_tokenized'] = df['text_punc'].apply(lambda x: tokenize(x))


# In[451]:


df.head()


# ## Remove Stop Words

# In[452]:


def remove_stopwords(text):
    sw = stopwords.words('english')
    temp = [i for i in text if i not in sw]
    return temp


# In[453]:


df['text_nonSW'] = df['text_tokenized'].apply(lambda x: remove_stopwords(x))


# In[454]:


df.head()


# ## Lemmatize

# In[455]:


wn = WordNetLemmatizer()
def Lemmatizing(text):
    temp = [wn.lemmatize(i) for i in text]
    return temp


# In[456]:


df['text_lemmatized'] = df['text_nonSW'].apply(lambda x: Lemmatizing(x))


# In[457]:


df.head()


# In[458]:


def arr_to_str(text):
    temp = ' '.join([i for i in text])
    return temp


# In[459]:


df['clean_text'] = df['text_lemmatized'].apply(lambda x: arr_to_str(x))


# In[460]:


df.head()


# ## TF - IDF Vectorize

# In[473]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[474]:


tf_idf_vec = TfidfVectorizer()


# In[475]:


x_Tf_idf = tf_idf_vec.fit_transform(df['clean_text'])


# In[476]:


df_Tf_idf = pd.DataFrame(x_Tf_idf.toarray())


# In[477]:


df_Tf_idf.columns = tf_idf_vec.get_feature_names()


# In[539]:


df_Tf_idf.head()


# In[540]:


df_Tf_idf = pd.concat([df['len_text'], df['punc_per'],df_Tf_idf], axis=1)


# ## Feature Engineering

# In[479]:


df.head()


# In[480]:


df['len_text'] = df['text'].apply(lambda x: len(x) - x.count(" "))


# In[504]:


def percentage_of_punc(text):
    temp = sum([1 for i in text if i in string.punctuation])
    return round(temp/(len(text)- text.count(' ') )*100, 2)


# In[505]:


df['punc_per'] = df['text'].apply(lambda x: percentage_of_punc(x))


# In[506]:


df.head()


# In[517]:


import matplotlib.pyplot as plt
import numpy as np 


# In[532]:


bins = np.linspace(0,200,40)
plt.hist(df[df['label'] == 'ham']['len_text'], bins=bins, label='ham', alpha = 0.5, normed=True)
plt.hist(df[df['label'] == 'spam']['len_text'], bins=bins, label='spam', alpha = 0.5, normed=True)
plt.legend()
plt.show()


# In[533]:


bins = np.linspace(0,50,40)
plt.hist(df[df['label'] == 'ham']['punc_per'], bins=bins, label='ham', alpha = 0.5, normed=True)
plt.hist(df[df['label'] == 'spam']['punc_per'], bins=bins, label='spam', alpha = 0.5, normed=True)
plt.legend()
plt.show()


# ## RandomForestClassifier

# In[534]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score


# In[542]:


cls = RandomForestClassifier()
k_fold = KFold(n_splits=5)
cross_val_score(cls, df_Tf_idf, df['label'], cv=k_fold, scoring='accuracy')


# In[ ]:




