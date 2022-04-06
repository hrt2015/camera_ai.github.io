import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Đọc dữ liệu
link= 'E:\\MINGTOM\\NEW\\ACTION_DATA\\'
test_df = pd.read_csv(link+"TEST.txt")
test2_df = pd.read_csv(link+"TEST2.txt")

X = []
y = []
no_of_timesteps = 10

dataset = test_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)
dataset = test2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)



X, y = np.array(X), np.array(y)
from keras.utils.np_utils import to_categorical
y = to_categorical(y, dtype="uint8")
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units= 2, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

#binary_crossentropy
model.summary()

model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("modeltest.h5")


