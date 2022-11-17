import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec

modelpath = 'models/'

# key vectors
with open(modelpath+'w2v_key_vecs.json','r') as f:
    w2v_key_vecs = json.load(f)
with open(modelpath+'lda_key_vecs.json','r') as f:
    lda_key_vecs = json.load(f)

# Vectorizer
with open(modelpath+'vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)

# Vectorize key vectors
w2v_vectorized_key_vecs = {}
lda_vectorized_key_vecs = {}
for key, vec in w2v_key_vecs.items():
    w2v_vectorized_key_vecs[key] = vectorizer.transform(vec)
for key, vec in lda_key_vecs.items():
    lda_vectorized_key_vecs[key] = vectorizer.transform(vec)

def identify(test_vec):
    vectorized_test_vec = vectorizer.transform(test_vec)

    w2v_cuisines = {}
    for key, vec in w2v_vectorized_key_vecs.items():
        A = vec
        B = vectorized_test_vec
        cossim = np.mean(cosine_similarity(A,B,dense_output=True))
        w2v_cuisines[key] = cossim

    lda_topics = {}
    for key, vec in lda_vectorized_key_vecs.items():
        A = vec
        B = vectorized_test_vec
        cossim = np.mean(cosine_similarity(A,B,dense_output=True))
        lda_topics[key] = cossim

    tot = sum(val for _,val in w2v_cuisines.items())
    w2v_cuisines = {key: round(100*(val/tot),2) for key,val in sorted(w2v_cuisines.items(), key=lambda item: -item[1]) if val > 0}
    w2v_cuisines['other'] = 0
    remove = []
    for key, val in w2v_cuisines.items():
        if val < 5:
            remove.append(key)
            w2v_cuisines['other'] += val
    for key in remove: w2v_cuisines.pop(key)

    tot = sum(val for _,val in lda_topics.items())
    lda_topics = {key: round(100*(val/tot),2) for key,val in sorted(lda_topics.items(), key=lambda item: -item[1]) if val > 0}
    lda_topics['other'] = 0
    remove = []
    for key, val in lda_topics.items():
        if val < 5:
            remove.append(key)
            lda_topics['other'] += val
    for key in remove: lda_topics.pop(key)

    return w2v_cuisines, lda_topics

##########################
# t-SNE Plot
w2v_model = Word2Vec.load(modelpath+'new_word_embedding_model.model')

def df_plot(recipe_name, ings, n_topwords):
    
    # Create embedding clusters for the ingredients
    embedding_clusters = []
    word_clusters = []
    for word in ings:
        try: # will only accept words the w2v model has seen
            embedding_clusters.append(w2v_model.wv[word])
            word_clusters.append(word)
        except:
            continue
    embedding_clusters = np.array(embedding_clusters)

    # load SVD model fit on total data then transform ingredient clusters
    with open(modelpath+'tsne_plot/svdmodel_{}.pkl'.format(n_topwords),'rb') as f:
        svdmodel = pickle.load(f)
    embeddings = np.array(svdmodel.transform(embedding_clusters))

    # create df with the average values for the whole recipe
    df = pd.DataFrame(columns=['x','y','label','distinction'])
    x = np.mean(embeddings[:,0])
    y = np.mean(embeddings[:,1])
    for embedding, ing in zip(embeddings,ings):
        df.loc[len(df.index)] = [embedding[0],embedding[1],ing,'ingredient']
    df.loc[len(df.index)] = [x,y,recipe_name,'title']

    return df
# ings = ['ground beef','ketchup','salt','hamburger buns',
# 'slices american cheese','mustard','garlic powder','worcestershire sauce']
# n_topwords = 10
# recipedf = df_plot('cheeseburger',ings,10)
# print(recipedf)

