import pandas as pd 
from ast import literal_eval
import pickle
import compress_pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

RELMODELPATH = 'streamlitapp/models/'
DATAPATH = '../data/' # edit this to where the data is saved

# load data
print('loading data')
df = pd.read_csv(DATAPATH+'newcleaned_trainfile.csv')
df.ingredients = [literal_eval(x) for x in df.ingredients]
df = df.drop(columns = ['Unnamed: 0'])
print('data loaded')

# Setup vectorizer inputs
string_ingr = [] # input strings (documents) for vectorizer
for ingr_list in df.ingredients:
    string_ingr.append(' '.join(ingr_list)) 

# load fitted vectorizer (~34 sec)
print('loading vectorizer and vectorizing data')
with open(RELMODELPATH+'vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)
vect_ingr = vectorizer.transform(string_ingr)
print('vectorizer ready')

# train LDA model
print('fitting LDA model')
lda = LatentDirichletAllocation(n_components = 20, random_state=0,n_jobs=6,verbose=3)
lda.fit(vect_ingr) # fit to the vectorized ingredients list
print('LDA model fit')

# Save Models
filename = 'lda_model_n20.gz'
with open(RELMODELPATH+filename,'wb') as f:
        compress_pickle.dump(lda, f, compression='gzip')
print('saved model to:',filename)