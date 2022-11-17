import pandas as pd 
from ast import literal_eval
import pickle
import compress_pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

modelpath = 'streamlitapp/models/'
datapath = '../data/'

# load data
print('loading data')
df = pd.read_csv(datapath+'newcleaned_trainfile.csv')
df.ingredients = [literal_eval(x) for x in df.ingredients]
df = df.drop(columns = ['Unnamed: 0'])
print('data loaded')

# load fitted vectorizer (~34 sec)
print('opening and fitting vectorizer')
with open(modelpath+'vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)
vect_ingr = vectorizer.transform(string_ingr)
vect_features = vectorizer.get_feature_names_out()
print('vectorizer ready')

# train LDA model
print('fitting LDA model')
lda = LatentDirichletAllocation(n_components = 20, random_state=0,n_jobs=6,verbose=3)
lda.fit(vect_ingr) # fit to the vectorized ingredients list
print('LDA model fit')

# Save Models
filename = 'lda_models/lda_model_n20.pkl'
with open(filename,'wb') as f:
        compress_pickle.dump(lda, f)
print('saved model to:',filename)