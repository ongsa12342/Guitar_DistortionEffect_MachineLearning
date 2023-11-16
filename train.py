import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd

SampleRate = 480000 #Hz

#Sustain Note
Sus_Clean, _ = librosa.load(f'data\SustainNote_Clean.wav',sr = SampleRate)
Sus_Distored, _ = librosa.load(f'data\SustainNote_Distorted.wav',sr = SampleRate)

#Chromatic
Chr_Clean, _ = librosa.load(f'data\Chromatic_Clean.wav',sr = SampleRate)
Chr_Distored, _ = librosa.load(f'data\Chromatic_Distorted.wav',sr = SampleRate)

#Major Chord
Maj_Clean, _ = librosa.load(f'data\MajorChord_Clean.wav',sr = SampleRate)
Maj_Distored, _ = librosa.load(f'data\MajorChord_Distorted.wav',sr = SampleRate)

#Minor Chord
Min_Clean, _ = librosa.load(f'data\MinorChord_Clean.wav',sr = SampleRate)
Min_Distored, _ = librosa.load(f'data\MinorChord_Distorted.wav',sr = SampleRate)

#Power Chord
Pow_Clean, _ = librosa.load(f'data\PowerChord_Clean.wav',sr = SampleRate)
Pow_Distored, _ = librosa.load(f'data\PowerChord_Distorted.wav',sr = SampleRate)

#Random
Ran_Clean, _ = librosa.load(f'data\Random_Clean.wav',sr = SampleRate)
Ran_Distored, _ = librosa.load(f'data\Random_Distorted.wav',sr = SampleRate)

#Solo
Solo_Clean, _ = librosa.load(f'data\Solo_Clean.wav',sr = SampleRate)
Solo_Distored, _ = librosa.load(f'data\Solo_Distorted.wav',sr = SampleRate)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.DataFrame({'Distorted':Chr_Clean,'Clean':Chr_Distored})

sample_size = 64
shifted_columns = [df['Clean'].shift(i).rename(f'Clean_-{i}') for i in range(1, sample_size)]
shifted_df = pd.concat(shifted_columns, axis=1)

# Combine the original DataFrame with the shifted DataFrame
df = df.join(shifted_df)

# Remove rows with NaN values
df = df.iloc[sample_size:-1]

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.33, random_state=42)

X_train = train_data.iloc[:,1:]
y_train = train_data[['Distorted']]


X_test = test_data.iloc[:,1:]
y_test = test_data[['Distorted']]

import tensorflow as tf

# Size of frame (in samples) that is fed to the model during training
frame = 64
# Chunk == sample
chunk = 1

hidden_units = 16

batch_size_para = 64

# Epochs during training
epochs_ = 64

tf.keras.backend.clear_session()
model = tf.keras.Sequential(name='NNguitar')
    
model.add(tf.keras.layers.Input(shape=(frame, chunk)))
model.add(tf.keras.layers.LSTM(hidden_units, activation='tanh', return_sequences=True, name='layer1'+'NNguitar'))
model.add(tf.keras.layers.LSTM(hidden_units, activation='tanh', return_sequences=False, name= 'layer2'+'NNguitar'))
model.add(tf.keras.layers.Dense(1)) 
 
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss= 'mean_absolute_error')

callback_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    
    batch_size=batch_size_para,
    shuffle=True,
    epochs=epochs_,
    callbacks = [callback_stop],
    validation_split = 0.15,
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('model.png')  # Change the file extension as needed

model.save('/NN_Chromatic_64epoch.h5')