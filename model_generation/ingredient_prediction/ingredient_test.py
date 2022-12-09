import tensorflow as tf
import spacy
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load Model
model = tf.keras.models.load_model('model.h5', compile=False)
thresh = 0.02       # Softmax Threshold for True output

# Open JSON Files --------------------------------------------------------------
f = open('json/ingredients_999.json')
ing_set = json.load(f)
ing_keys = list(ing_set.keys())
f.close()

f = open('json/title_tokens_999.json')
title_toks = json.load(f)
title_toks_keys = list(title_toks.keys())
f.close()

# Initialize Spacy--------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# nlp = spacy.load("en_core_web_sm")

INGREDIENT_STOPWORDS = [
                'teaspoon','teaspoons', 'tsp',
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

TITLE_STOPWORDS = [
                "recipe", "'", "&", "'s", "/", "-", "#", "w/", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"
                ]

print("Configuration Complete")

test_df = pd.read_pickle('df_test.pkl')
n = len(test_df)
print(n, 'test recipes')
test_df['x'] = test_df['x'].apply(lambda x: np.array(x))
test_df['y'] = test_df['y'].apply(lambda y: np.array(y))
X = np.array(test_df['x'].tolist())
y = np.array(test_df['y'].tolist())
X = X.astype(float)
y = y.astype(float)
del test_df

print("Test DataFrame Loaded")

preds = model.predict(X)
print(preds.shape)

print("Evlauating Performance Metrics")

tps = [0 for i in range(1000)]
tns = [0 for i in range(1000)]
fps = [0 for i in range(1000)]
fns = [0 for i in range(1000)]

pre = [0 for i in range(1000)]
rec = [0 for i in range(1000)]
f1s = [0 for i in range(1000)]

for i in range(preds.shape[0]):
    pred = preds[i]
    true = y[i]
    for j in range(pred.shape[0]):
        if pred[j] >= thresh:
            if true[j] == 1:
                # True positive
                tps[j] += 1
            else:
                # False positive
                fps[j] += 1
        else:
            if true[j] == 1:
                # False negative
                fns[j] += 1
            else:
                # True Negative
                tns[j] += 1

# Calculate Precision, Recall, and F1
for i in range(len(tps)):
    if tps[i] == 0:
        pre[i] = 0
        rec[i] = 0
        f1s[i] = 0
    else:
        pre[i] = tps[i] / (tps[i] + fps[i])
        rec[i] = tps[i] / (tps[i] + fns[i])
        f1s[i] = 2 * (pre[i] * rec[i]) / (pre[i] + rec[i])


preds_bin = (preds >= thresh).astype(int)

i = 0
n_pred = 0
n_true = 0
for i in range(preds_bin.shape[0]):
    n_pred += preds_bin[i].sum()
    n_true += y[i].sum()
    i += 1

# Print Out Metrics
print('{} true ingredients ({} per recipe)'.format(n_true, np.round(n_true / n, 2)))
print('{} predicted ingredients ({} per recipe)'.format(n_pred, np.round(n_pred / n, 2)))
print('True Positives:', sum(tps))
print('True Negatives:', sum(tns))
print('False Positives:', sum(fps))
print('False Negatives:', sum(fns))
print('{} correctly predicted ingredients per recipe (on average)'.format(np.round(sum(tps) / n, 2)))
print('Overall Accuracy: {}'.format(np.round((sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns)), 3)))
p = sum(tps) / (sum(tps) + sum(fps))
r = sum(tps) / (sum(tps) + sum(fns))
f1 = 2 * (p * r) / (p + r)
print('Aggregate - Precision: {}, Recall: {}, F1: {}'.format(np.round(p, 3), np.round(r, 3), np.round(f1, 3)))
p = np.divide(np.array(tps), np.add(np.array(tps), np.array(fps)), out=np.zeros_like(np.array(tps, dtype=np.float64)), where= np.add(np.array(tps), np.array(fps))!=0).mean()
r = np.divide(np.array(tps), np.add(np.array(tps), np.array(fns)), out=np.zeros_like(np.array(tps, dtype=np.float64)), where= np.add(np.array(tps), np.array(fns))!=0).mean()
f1 = 2 * (p * r) / (p + r)
print('Mean Classwise - Precision: {}, Recall: {}, F1: {}'.format(np.round(p, 2), np.round(r, 2), np.round(f1, 2)))

# Plot Performance for Top 50 Ingredients
a = []      # Ingredients
b = []      # True Predictions (Log10)
c = []      # Total Occurrences (log10)
d = []      # Total Predictions (log10)
for arg in range(50):
    if arg == 0:
        a.append('Other')
    else:
        a.append(ing_keys[arg-1])
    b.append(np.log10(tps[arg]))
    c.append(np.log10(tps[arg] + fns[arg]))
    d.append(np.log10(tps[arg] + fps[arg]))

# Subplots
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
# f.suptitle('Model Performance for 50 Most Frequent Ingredients')
ax1.set_title('Model Performance for 50 Most Frequent Ingredients')
plt.subplots_adjust(bottom=0.3, hspace=0.1)
plt.xticks(rotation=90)
ax1.bar(a, c, color='gray', alpha=0.5, label='Total Occurences')
ax1.bar(a, b, color='green', alpha=0.6, label='True Predictions')
ax1.legend()
ax1.set_ylabel('Count (log10)')
ax2.bar(a, d, color='red', alpha=0.3, label='Total Predictions')
ax2.bar(a, b, color='green', alpha=0.6, label='True Predictions')
ax2.set_ylabel('Count (log10)')
ax2.legend()
plt.show()
