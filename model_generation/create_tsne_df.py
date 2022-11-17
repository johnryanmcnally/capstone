import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import pickle
from sklearn.decomposition import TruncatedSVD
import networkx as nx
import itertools
from collections import Counter
import json
from ast import literal_eval

# load w2v model if already trained
relmodelpath = 'streamlit_app/models/'
w2v_model = Word2Vec.load(relmodelpath+'new_word_embedding_model.model')

def df_plot(keys, n_topwords):
    max_topwords = 30
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in w2v_model.wv.most_similar(word, topn=n_topwords):
            words.append(similar_word)
            embeddings.append(w2v_model.wv[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    perp = (n_topwords*len(keys))**.5 #st.slider('Perplexity',1,50,10)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print(n,m,k)
    tsne_model_en_2d = TSNE(perplexity=perp, n_components=2, init='pca', n_iter=3500, random_state=32)
    svd = TruncatedSVD(n_components=2, n_iter=5, random_state=32)
    embeddings_en_2d = np.array(svd.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    with open('streamlit_app/models/tsne_plot/svdmodel_{}.pkl'.format(n_topwords), 'wb') as f:
        pickle.dump(svd, f)

    colors = cm.nipy_spectral(np.linspace(0, .95, len(keys)))
        
    df = pd.DataFrame(columns=['x','y','color','label','word'])
    for label, embeddings, words, color in zip(keys, embeddings_en_2d, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        
        for i in range(len(x)):
            df.loc[len(df.index)] = [x[i],y[i],color,label,words[i]]

    return df

keys = ['american','mexican',
        'italian','polish','german','spanish','french','greek','english','portuguese',
        'indian','chinese','vietnamese','thai','japanese','korean',
       ]
n_topwords = [5,10,15,20,25,30]
# n_topwords = [10]

for n in n_topwords:
    df = df_plot(keys, n)
    df.to_csv('streamlit_app/models/tsne_plot/tsnedf_{}.csv'.format(n))

######################################################################################
## create common ingredient pairs df
print('loading df')
df = pd.read_csv('C:/Users/jrmcn/MADS/MADS_Capstone/data/newcleaned_trainfile.csv')
df.ingredients = [literal_eval(x) for x in df.ingredients]
df = df.drop(columns = ['Unnamed: 0'])
print('df loaded')
# create complete edge set
edges = []
for ingr_list in df.ingredients:
    for subset in itertools.combinations(ingr_list,2):
         edges.append(subset)

# calculate edge weights
weighted_edges = Counter()
weighted_edges.update(edges)
print('edges calculated')

most_common = [x[0] for x in weighted_edges.most_common(100000)]
with open('streamlit_app/models/common_pairs.json','w') as f:
    json.dump(most_common, f)
print('model saved')



