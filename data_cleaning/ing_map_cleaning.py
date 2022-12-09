import pandas as pd
import numpy as np
from ast import literal_eval
import json

# load manual ingredient mapping
with open('data_cleaning/topkey_mapping.json','r') as f:
    sp = json.load(f)

# load data from parsed json
print('loading data')
DATAPATH = '../data/'  # Change to be the folder where parsed_trainfile.csv is located
df = pd.read_csv(DATAPATH+'parsed_trainfile.csv')
df.ingredients = [literal_eval(x) for x in df.ingredients]
df = df.drop(columns = ['Unnamed: 0'])
print('data loaded')

# Clean ingredients further with manually observed tokens
stopwords = ['chopped','diced','sliced','minced','10','oz','roughly','cut','half',10,"'10'", 'grams', 'slices','peeled','divided','g','t','','.','x','cubed','trimmed','sm','lg','possibly',
'ml','slice','grated']

def clean_ingredients(ingr_list):
    new_words = []
    for word in ingr_list:
        new_ing = []
        split = word.split()
        for subword in split:
            if subword.lower() not in stopwords:
                new_ing.append(subword.lower())
        new_ing = ' '.join(new_ing)

        replaced = False
        for key, vals in sp.items():
            if new_ing in vals:
                new_words.append(key)
                replaced = True

        if replaced == False:
            new_words.append(new_ing)
    return new_words
print('cleaning ingredients')
df.ingredients = df.ingredients.apply(clean_ingredients)
df.to_csv(DATAPATH+'newcleaned_trainfile.csv')
print('cleaning complete and saved to',DATAPATH+'newcleaned_trainfile.csv')
