import json
import re
import spacy


FILENAME = 'data/train.json' # Change this to be the JSON from the Recipe1M+ dataset
f = open(FILENAME)
recipes = json.load(f)
f.close()

print(len(recipes), "Recipes Loaded")

f = open('ingredient_map.json')
ing_map = json.load(f)
f.close()

print(len(ing_map), "mappings Loaded")

ing_dict = {}       # Track Reduced Ingredients and Frequency
# Counters
i = 0               # 10k Recipes Processed
j = 0               # 100k Recipes Processed
k = 0

# Set Up Spacy
# gpu = spacy.prefer_gpu()
# print('GPU:', gpu)
nlp = spacy.load("en_core_web_sm")
STOPWORDS = [   'teaspoon','teaspoons', 'tsp',
                'cup', 'cups', 'c.',
                'tablespoon', 'tablespoons', 'tbsp', 'tbs',
                'ounce', 'ounces',
                'quart', 'quarts',
                ' ',
                'lb', 'lbs', 'pound',
                'pinch', 'pinches',
                'finely', 'coarsely', 'thinly', 'to', 'taste',
                'can', 'cans', 'pkg', 'package', 'packages', 'jar',
                'sifted', 'melted', 'softened',
                'small', 'medium', 'large',
                'firmly', 'packed'
                # 'chopped', 'minced', 'sliced', 'diced'       #   *** Controversial ***
            ]


for recipe in recipes:
    ingredients = recipe['ingredients']
    for ingredient in ingredients:
        text = ingredient['text']
        text = re.sub(r'\([^)]*\)', '', text)           # Remove text within parenthesis
        doc = nlp(text)
        ing = []
        for token in doc:
            if token.pos_ not in ['NUM', 'PUNCT', 'CCONJ']:
                if token.lower_ not in STOPWORDS:
                    ing.append(token.lower_)
        # Reassemble Ingredient Name and Add to Dictionary
        ing = ' '.join(ing).strip()
        # If key in mapping, replace with reduced version
        if ing in ing_map.keys():
          ing = ing_map[ing]
        if ing in ing_dict:
            ing_dict[ing] += 1
        else:
            ing_dict[ing] = 1

    # Print Progress Status to Command Line
    i += 1
    if i > 10000:
        i = 0
        j += 1
        print("10k Recipes Processed")
        if j >= 10:
            j = 0
            k += 1
            print("100k Recipes Processed")
            out_json = json.dumps(dict(sorted(ing_dict.items(), key=lambda x: -x[1])[:999]))
            with open("ingredients_999_reduced_" + str(k) + "00k.json", "w") as outfile:
                outfile.write(out_json)

print("Done")
# print(ing_dict)
out_json = json.dumps(dict(sorted(ing_dict.items(), key=lambda x: -x[1])))
with open("ingredients_reduced_train.json", "w") as outfile:
    outfile.write(out_json)
out_json = json.dumps(dict(sorted(ing_dict.items(), key=lambda x: -x[1])[:999]))
with open("ingredients_999_reduced_train.json", "w") as outfile:
    outfile.write(out_json)
