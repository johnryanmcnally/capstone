import tensorflow as tf
import pandas as pd
import numpy as np
import re
import string
import einops
import streamlit as st
import cv2

modelpath = 'streamlitapp/models/'
def get_model():
    IMAGE_SHAPE = (224, 224, 3)     # Mobilenet Image Embedder Compatibility
    DF_PATH = modelpath+"df.csv"
    WEIGHTS_PATH = modelpath+"cp-0022.ckpt"

    def standardize(s):
        s = tf.strings.lower(s)
        s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
        s = tf.strings.regex_replace(s, f'\d+', '')
        s = tf.strings.join(['startSeq', s, 'endSeq'], separator=' ')
        return s

    mobilenet = tf.keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        include_preprocessing=True,
        weights='imagenet'
    )

    # Initialize Tokenizer with vocabulary of 1,000
    df = pd.read_csv(DF_PATH)
    vocab_size = 1000
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=standardize,
        ragged=True
    )
    tokenizer.adapt(np.asarray(df['Title']).astype(str))

    # Create Tensorflow-compatible string-lookup tables
    # to map indices to words
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)


    # Model layer Definitions
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
            seq = self.token_embedding(seq)

            x = tf.range(tf.shape(seq)[1])
            x = x[tf.newaxis, :]
            x = self.pos_embedding(x)

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

        def call(self, x):
            x = self.dense(x)
            return x

    output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', 'startSeq'))

    # Model Construction
    class Captioner(tf.keras.Model):
        @classmethod
        def add_method(cls, fun):
            setattr(cls, fun.__name__, fun)
            return fun

        def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1, units=256, max_length=50, num_heads=1, dropout_rate=0.1):
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

    @Captioner.add_method
    def simple_gen(self, image, temperature=1):
        initial = self.word_to_index([['startSeq']])    # Batch, sequence
        img_features = self.feature_extractor(image[tf.newaxis, ...])
        tokens = initial
        for n in range(50):
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

    # Cache This ********************************************
    model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                    units=225, dropout_rate=0.5, num_layers=2, num_heads=2)
    model.load_weights(WEIGHTS_PATH)
    # Cache This ********************************************
    return model