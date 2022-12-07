# Training Script for Recipe Recognition Model
# Image --> Recipe Title
# Adapted from Tensorflow's "Image captioning with visual attention" tutorial
# https://www.tensorflow.org/tutorials/text/image_captioning
# Layer construction taken directly from tutorial with slight modifications

import tensorflow as tf
import pandas as pd
import numpy as np
import re
import string
import collections
import einops
import matplotlib.pyplot as plt

IMAGE_SHAPE = (224, 224, 3)     # MobileNet Image Shape
# IMAGE_SHAPE = (299, 299, 3)     # Inception Image Shape
BATCH_SIZE = 16
MAX_LENGTH = 50                     # Maximum number of tokens in recipe name
train = True                       # Boolean to either train (True) or skip to inferencing (False)
CKPT_PATH = 'ckpts/cp-0022.ckpt'    # Weights to load if not training

# Training Dataset: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images
CSV_PATH = "epicurious_dataset.csv"         # Path to CSV w/ Title & Image Name Columns
IMG_PATH = "EpicImages/EpicImages/*.jpg"    # Folder with all images
TEST_PATH = "mac.jpg"                # Test Image for Attention Maps & Epoch Print Outs

# Util Functions
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])
    # img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def standardize(s):
  s = tf.strings.lower(s)
  s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
  s = tf.strings.regex_replace(s, f'\d+', '')
  s = tf.strings.join(['startSeq', s, 'endSeq'], separator=' ')
  return s

# Image Feature Extractor
mobilenet = tf.keras.applications.MobileNetV3Small(
  input_shape=IMAGE_SHAPE,
  include_top=False,
  include_preprocessing=True,
  weights='imagenet'
)

# inception = tf.keras.applications.inception_v3.InceptionV3(
#     input_shape=IMAGE_SHAPE,
#     include_top=False,
#     weights='imagenet'
# )

# Parse Text + Build Tokenizers and StringLookups
df = pd.read_csv(CSV_PATH)
vocab_size = 1000
tokenizer = tf.keras.layers.TextVectorization(
  max_tokens=vocab_size,
  standardize=standardize,
  ragged=True
)
# Initialize tokenizer to column of recipe titles
tokenizer.adapt(np.asarray(df['Title']).astype(str))
# StringLookups to convert back and forth from index to word
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

ds = tf.data.Dataset.list_files(IMG_PATH)
print('Length of Dataset:', len(ds))
ds = ds.shuffle(1000, seed=0)
train_size = int(0.8 * len(ds))
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)


def producePair(fpath):
    # Extract image title and image file from file path
    try:
        image = 0
        image = load_image(fpath)
        label = tf.strings.split(fpath, '\\')[-1]
        label = tf.strings.split(label, '.jpg')[0]
        label = tf.strings.regex_replace(label, f'-', ' ')
    except:
        # Soft fail
        print("Error parsing", fpath)
        image = 0
        label = 'null'
    return image, label

def prepare_txt(imgs, txts):
    # Offset input tokens and labels (next token)
    tokens = tokenizer(txts)
    input_tokens = tokens[..., :-1]
    label_tokens = tokens[..., 1:]
    return (imgs, input_tokens), label_tokens

def to_tensor(inputs, labels):
    # Convert tokens to tensors
    (images, in_tok), out_tok = inputs, labels
    return (images, in_tok.to_tensor()), out_tok.to_tensor()

# Build datasets, filter faulty records, and batch
train_ds = (train_ds
        .map(producePair)
        .filter(lambda x, y: tf.math.not_equal(y, tf.constant('null')))
        .apply(tf.data.experimental.ignore_errors(log_warning=True))
        .batch(BATCH_SIZE)
        .map(prepare_txt, tf.data.AUTOTUNE)
        .map(to_tensor, tf.data.AUTOTUNE)
        )

val_ds = (val_ds
        .map(producePair)
        .filter(lambda x, y: tf.math.not_equal(y, tf.constant('null')))
        .apply(tf.data.experimental.ignore_errors(log_warning=True))
        .batch(BATCH_SIZE)
        .map(prepare_txt, tf.data.AUTOTUNE)
        .map(to_tensor, tf.data.AUTOTUNE)
        )

# Build Model-------------------------------------------------------------------
class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)

        self.token_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=depth,
        mask_zero=True)

        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq = self.token_embedding(seq) # (batch, seq, depth)

        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.pos_embedding(x)  # (1, seq, depth)

        return self.add([seq,x])

class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        # Use Add instead of + so the keras mask propagates through.
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn = self.mha(query=x, value=x, use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, y):
        attn, attention_scores = self.mha( query=x, value=y, return_attention_scores=True)

        self.last_attention_scores = attention_scores

        x = self.add([x, attn])
        return self.layernorm(x)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(units=2*units, activation='relu'),
        tf.keras.layers.Dense(units=units),
        tf.keras.layers.Dropout(rate=dropout_rate),
        ])

        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.seq(x)
        return self.layernorm(x)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    def call(self, inputs, training=False):
        in_seq, out_seq = inputs
        # Text inputs
        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        out_seq = self.ff(out_seq)
        return out_seq

class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', 'startSeq'), **kwargs):
        super().__init__()

        self.dense = tf.keras.layers.Dense(units=tokenizer.vocabulary_size())
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens

        self.bias = None

    def adapt(self, ds):
        counts = collections.Counter()
        vocab_dict = {name: id for id, name in enumerate(self.tokenizer.get_vocabulary())}
        i = 0
        for tokens in ds:
            counts.update(np.array(tokens).flatten())

        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())

        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0

        total = counts_arr.sum()
        p = counts_arr/total
        p[counts_arr==0] = 1.0
        log_p = np.log(p)  # log(1) == 0

        entropy = -(log_p*p).sum()
        self.bias = log_p
        self.bias[counts_arr==0] = -1e9

    def call(self, x):
        x = self.dense(x)
        if train:
            return tf.nn.bias_add(x, self.bias)
        return x

output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', 'startSeq'))
output_layer.adapt(train_ds.map(lambda inputs, labels: labels))

class Captioner(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1, units=256, max_length=MAX_LENGTH, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
        self.index_to_word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)
        self.seq_embedding = SeqEmbedding(vocab_size=tokenizer.vocabulary_size(), depth=units, max_length=max_length)
        self.decoder_layers = [
        DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate) for n in range(num_layers)
        ]
        self.output_layer = output_layer

@Captioner.add_method
def call(self, inputs):
    image, txt = inputs

    if image.shape[-1] == 3:
        image = self.feature_extractor(image)
    image = einops.rearrange(image, 'b h w c -> b (h w) c')

    if txt.dtype == tf.string:
        txt = tokenizer(txt)
    txt = self.seq_embedding(txt)

    for dec_layer in self.decoder_layers:
        txt = dec_layer(inputs=(image, txt))

    txt = self.output_layer(txt)
    return txt

model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                  units=225, dropout_rate=0.5, num_layers=2, num_heads=2)

@Captioner.add_method
def simple_gen(self, image, temperature=1):
    initial = self.word_to_index([['startSeq']])    # Batch, sequence
    img_features = self.feature_extractor(image[tf.newaxis, ...])
    tokens = initial
    for n in range(MAX_LENGTH):
        preds = np.array(self((img_features, tokens)))
        preds = preds[:,-1,:]
        if temperature == 0:
            next = tf.argmax(preds, axis=-1)[:, tf.newaxis]
        else:
            next = tf.random.categorical(preds / temperature, num_samples=1)
        tokens = tf.concat([tokens,next], axis=1)

        if next[0] == self.word_to_index('endSeq'):
            break
    words = index_to_word(tokens[0, 1:-1])
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    return result

# Losses and Metrics
def masked_loss(labels, preds):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)
    mask = (labels != 0) & (loss < 1e8)
    mask = tf.cast(mask, loss.dtype)
    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_acc(labels, preds):
    mask = tf.cast(labels!=0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match*mask) / tf.reduce_sum(mask)
    return acc

class GenerateText(tf.keras.callbacks.Callback):
    def __init__(self):
        # image_path = 'taco.jpg'
        image_path = TEST_PATH
        self.image = load_image(image_path)

    def on_epoch_end(self, epochs=None, logs=None):
        # Print Out Predicted Title of Test Image Every Epoch
        print()
        print()
        for t in (0.0, 0.5, 1.0):
            result = model.simple_gen(self.image, temperature=t)
            print(result.numpy().decode())
        print()

callbacks = [   GenerateText(),
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(filepath='ckpts/cp-{epoch:04d}.ckpt', save_weights_only=True, verbose=1)]

# Train the model -- Comment out to skip to inferencing
if train:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=masked_loss,
        metrics=[masked_acc]
    )

    history = model.fit(
        train_ds.repeat(),
        steps_per_epoch=680,
        validation_data=val_ds.repeat(),
        validation_steps=170,
        epochs=1,
        callbacks=callbacks
    )

# Inferencing + Attention Maps

# Load Saved Weights (if necessary)
if not train:
    model.load_weights(CKPT_PATH)
im = load_image(TEST_PATH)
result = model.simple_gen(im, temperature=0.0)
result = result.numpy().decode()
str_tokens = result.split()
str_tokens.append('endSeq')
attn_maps = [layer.last_attention_scores for layer in model.decoder_layers]
attention_maps = tf.concat(attn_maps, axis=0)
attention_maps = einops.reduce(
    attention_maps,
    'batch heads sequence (height width) -> sequence height width',
    height=7, width=7,
    reduction='mean')
einops.reduce(attention_maps, 'sequence height width -> sequence', reduction='sum')

def plot_attention_maps(image, str_tokens, attention_map):
    fig = plt.figure(figsize=(16, 9))

    len_result = len(str_tokens)

    titles = []
    for i in range(len_result):
      map = attention_map[i]
      grid_size = max(int(np.ceil(len_result/2)), 2)
      ax = fig.add_subplot(3, grid_size, i+1)
      titles.append(ax.set_title(str_tokens[i]))
      img = ax.imshow(image)
      ax.imshow(map, cmap='gray', alpha=0.6, extent=img.get_extent(),
                clim=[0.0, np.max(map)])

    plt.tight_layout()
    plt.show()

plot_attention_maps(im/255, str_tokens, attention_maps)
