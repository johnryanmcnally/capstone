import pandas as pd
import numpy as np
import json
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import compress_pickle


# load w2v model
relmodelpath = 'streamlit_app/models/'
w2v_model = Word2Vec.load(relmodelpath+'new_word_embedding_model.model')
keys = ['american','mexican',
        'italian','polish','german','spanish','french','greek','english','portuguese',
        'indian','chinese','vietnamese','thai','japanese','korean',
    ]
def w2v_keyvectors(model, keys, n_topwords=30):
    key_vecs = {}
    for key in keys:
        key_vecs[key] = model.wv.most_similar(key, topn=n_topwords)
    key_vectors = {}
    for key, vals in key_vecs.items():
        values = []
        for val in vals:
            values.append(val[0])
        key_vectors[key] = values

    with open('streamlit_app/models/w2v_key_vecs.json','w') as f:
        json.dump(key_vectors,f)
# w2v_keyvectors(w2v_model,keys)

# load lda model
# with open(relmodelpath+'lda_model_n20.pkl','rb') as f:
#     lda = pickle.load(f)

lda = compress_pickle.load(relmodelpath+'compressed_lda_model_n20', compression='gzip')
# compress_pickle.dump(lda,relmodelpath+'compressed_lda_model_n20', compression='gzip')

# load vectorizer
with open(relmodelpath+'vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)
feature_names = vectorizer.get_feature_names_out()

def lda_keyvectors(model, feature_names, n_topwords=30):
    topic_names = {
    0:'bread',
    1:'soup',
    2:'italian',
    3:'middle eastern',
    4:'pie',
    5:'mayo-based salad',
    6:'pizza',
    7:'potato',
    8:'salad',
    9:'bbq',
    10:'fruity dessert',
    11:'rich dessert',
    12:'cake',
    13:'asian',
    14:'banana dessert',
    15:'cajun',
    16:'braised meat',
    17:'punch',
    18:'mexican',
    19:'sweet pastry'
    }

    topic_vectors = {}
    
    for idx, val in topic_names.items():
        topic = model.components_[idx]
        term_list = [feature_names[i] for i in topic.argsort()[:-n_topwords - 1:-1]]
        topic_vectors[val]=term_list
    
    with open('streamlit_app/models/lda_key_vecs.json','w') as f:
        json.dump(topic_vectors,f)

lda_keyvectors(lda, feature_names)