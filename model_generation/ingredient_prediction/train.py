import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# Training Weights (Wr) for incentivising rarer ingredient predictions
# wts = np.array([1 for x in range(1000)]) / 1.                     # Wr = 1
# wts = np.array([1000 + 2*x for x in range(1000)]) / 1000.       # Wr = 3
wts = np.array([1000 + 4*x for x in range(1000)]) / 1000.       # Wr = 5

# Custom loss apply weight for positive classes and rare ingredients
def weighted_binary_crossentropy(target, output):
    # transform back to logits
    _epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=1000)*wts
    return tf.reduce_mean(loss, axis=-1)


# Load Train and Validation DataFrames
train_df = pd.read_pickle('df_train.pkl')
train_df['x'] = train_df['x'].apply(lambda x: np.array(x))
train_df['y'] = train_df['y'].apply(lambda y: np.array(y))
X = np.array(train_df['x'].tolist())
y = np.array(train_df['y'].tolist())
X = X.astype(float)
y = y.astype(float)
del train_df
val_df = pd.read_pickle('df_val.pkl')
val_df['x'] = val_df['x'].apply(lambda x: np.array(x))
val_df['y'] = val_df['y'].apply(lambda y: np.array(y))
X_val = np.array(val_df['x'].tolist())
y_val = np.array(val_df['y'].tolist())
X_val = X_val.astype(float)
y_val = y_val.astype(float)
del val_df

print("Dataframes loaded")


# Model Construction
input = Input(shape=(1000, ))
dense1 = Dense(128, activation='relu')(input)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
output = Dense(1000, activation='softmax')(dense3)
model = Model(input, output)

model.compile(loss=weighted_binary_crossentropy, optimizer='adam')

model.summary()

history = model.fit(X, y, epochs=10, batch_size=512, validation_data=(X_val, y_val))
model.save('model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.show()
