import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Đọc dữ liệu
link= 'D:\\KLTN\\NEW\\ACTION_DATASET_CSV\\'
testtrainvaytay500_df=pd.read_csv(link+"TESTTRAINVAYTAY2000.txt")
testtraindibo500_df=pd.read_csv(link+"TESTTRAINDIBO2000.txt")
testtrainuongnuoc500_df=pd.read_csv(link+"TESTTRAINUONGNUOC2000.txt")
testtrainngoixuong500_df=pd.read_csv(link+"TESTTRAINNGOIXUONG2000.txt")
testtraindachan500_df=pd.read_csv(link+"TESTTRAINDACHAN2000.txt")

X = []
y = []
no_of_timesteps = 200

dataset = testtrainvaytay500_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)
dataset = testtraindibo500_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)
dataset = testtrainuongnuoc500_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)
dataset = testtrainngoixuong500_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(3)
dataset = testtraindachan500_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(4)

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
model.add(Dense(units= 5, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")
#categorical_crossentropy >2 class
#binary_crossentropy =2 class
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=128,validation_data=(X_test, y_test))
model.save("model_test_check500.h5")
model.summary()


