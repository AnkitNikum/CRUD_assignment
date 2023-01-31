import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
import string
from num2words import num2words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import math

df = pd.read_csv('news.csv')
df.dropna(inplace=True)
df.reset_index(inplace=True)
train, test = train_test_split(df, test_size=0.2)
train.reset_index(inplace=True)
test.reset_index(inplace=True)

from nltk.corpus import stopwords
sw_nltk = stopwords.words('english')

def stopword_fun(x):
    words = [word for word in x['content'].split() if word.lower() not in sw_nltk]
    text = " ".join(words)
    text = "".join([i for i in text if i not in string.punctuation])
    words = [WordNetLemmatizer().lemmatize(word) for word in text.split(' ')]
    text = ''
    for word in words:
        
        if word.isnumeric():
            text  = text + " " +num2words(float(word))
        elif len(word) >2:
            text =  text + " " + word
    words = [word for word in text.split() if word.lower() not in sw_nltk]    
    text = " ".join(words)
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.lower()
    return text
def stopword_fun_title(x):
    words = [word for word in x['title'].split() if word.lower() not in sw_nltk]
    text = " ".join(words)
    text = "".join([i for i in text if i not in string.punctuation])
    words = [WordNetLemmatizer().lemmatize(word) for word in text.split(' ')]
    text = ''
    for word in words:
        
        if word.isnumeric():
            text  = text + " " +num2words(float(word))
        elif len(word) >2:
            text =  text + " " + word
    words = [word for word in text.split() if word.lower() not in sw_nltk]    
    text = " ".join(words)
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.lower()
    return text
def preprocess_query(x):
    words = [word for word in x.split() if word.lower() not in sw_nltk]
    text = " ".join(words)
    text = "".join([i for i in text if i not in string.punctuation])
    words = [WordNetLemmatizer().lemmatize(word) for word in text.split(' ')]
    text = ''
    for word in words:
        
        if word.isnumeric():
            text  = text + " " +num2words(float(word))
        elif len(word) >2:
            text =  text + " " + word
    words = [word for word in text.split() if word.lower() not in sw_nltk]    
    text = " ".join(words)
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.lower()
    return text

content = train.apply(stopword_fun,axis=1)

title = train.apply(stopword_fun_title,axis=1)

processed_text = []
processed_title = []
print(len(content),len(title))

for j,i in enumerate(content):
   
    processed_text.append(word_tokenize(i))
    processed_title.append(word_tokenize(title[j]))
DF = {}
for i in range(len(content)):
    tokens = processed_text[i]
    for w in tokens:
        if w in DF.keys():
            DF[w].add(i)
        else:
            DF[w] = {i}
   
    tokens = processed_title[i]
    for w in tokens:
        if w in DF.keys():
            DF[w].add(i)
        else:
            DF[w] = {i}

for i in DF:
    DF[i] = len(DF[i])
total_vocab_size = len(DF)

total_vocab = [x for x in DF]
def doc_freq(word):
    c = 0
    if word in DF.keys():
        c = DF[word]
    else:
        pass
    return c

doc = 0

tf_idf = {}
N = len(content)

for i in range(N):
    
    tokens = processed_text[i]
    
    counter = Counter(tokens + processed_title[i])
    words_count = len(tokens + processed_title[i])
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df_val = doc_freq(token)
        idf = np.log((N+1)/(df_val+1))
        
        tf_idf[doc, token] = tf*idf

    doc += 1

doc = 0

tf_idf_title = {}

for i in range(N):
    
    tokens = processed_title[i]
    counter = Counter(tokens + processed_text[i])
    words_count = len(tokens + processed_text[i])

    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df_val = doc_freq(token)
        idf = np.log((N+1)/(df_val+1)) #numerator is added 1 to avoid negative values
        
        tf_idf_title[doc, token] = tf*idf

    doc += 1

alpha = 0.7
for i in tf_idf:
    tf_idf[i] *= alpha
for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]

def matching_score(k, query):
    preprocessed_query = preprocess_query(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("Matching Score")
    print("\nQuery:", query)
    print("")
    print(tokens)
    
    query_weights = {}

    for key in tf_idf:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")
    
    l = []
    
    for i in query_weights[:k]:
        l.append(i[0])
    
    print(l)

matching_score(10,test['content'][0])

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass
def gen_vector(tokens):

    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = math.log((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q

def cosine_similarity(k, query):
    print("Cosine Similarity")
    preprocessed_query = preprocess_query(query)
    tokens = word_tokenize(str(preprocessed_query))
    
    print("\nQuery:", query)
    print("")
    print(tokens)
    
    d_cosines = []
    
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
       
    out = np.array(d_cosines).argsort()[-k:][::-1]
    
    print("")
    
    print(out)
    
cosine_similarity(10,test['content'][0])