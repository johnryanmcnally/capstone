import pandas as pd
import numpy as np
from time import time
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from ast import literal_eval
import multiprocessing
from gensim.models import Word2Vec

# change path to location where data is stored
DATAPATH = '../data/'
RELMODELPATH = 'streamlitapp/models/'

print('loading data')
df = pd.read_csv(DATAPATH+'newcleaned_trainfile.csv')
df.ingredients = [literal_eval(x) for x in df.ingredients]
df = df.drop(columns = ['Unnamed: 0'])

df['titlelist'] = [str(row).split() for row in df.title.values]
df['words'] = df.ingredients + df.titlelist
print('data loaded')

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=200,
                     window=10,
                     vector_size=3000,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1,
                     epochs=30)
t = time()
w2v_model.build_vocab(df.words, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# Train W2V model (~6 min)
t = time()
w2v_model.train(df.words, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# save model
w2v_model.save(RELMODELPATH+'new_word_embedding_model.model')
print('model trained and saved to', RELMODELPATH+'new_word_embedding_model.model')