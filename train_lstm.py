import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Đọc dữ liệu
link= 'E:\\MINGTOM\\NEW\\ACTION_DATASET_CSV\\'
vaytay_df = pd.read_csv(link+"VAYTAY.txt")
votay_df = pd.read_csv(link+"VOTAY.txt")
votay2_df = pd.read_csv(link+"VOTAY2.txt")
lacnguoi_df = pd.read_csv(link+"LACNGUOI.txt")
daubung_df = pd.read_csv(link+"DAUBUNG.txt")
nhucdau_df = pd.read_csv(link+"NHUCDAU.txt")
dibo_df = pd.read_csv(link+"DIBO.txt")
texiu_df = pd.read_csv(link+"TEXIU.txt")
texiu2_df = pd.read_csv(link+"TEXIU2.txt")
daulung_df = pd.read_csv(link+"DAULUNG.txt")
dauchan_df = pd.read_csv(link+"DAUCHAN.txt")
dautay_df = pd.read_csv(link+"DAUTAY.txt")
dauco_df = pd.read_csv(link+"DAUCO.txt")
uongnuoc_df = pd.read_csv(link+"UONGNUOC.txt")
dachankard_df = pd.read_csv(link+"DACHANKARD.txt")
dachankard2_df = pd.read_csv(link+"DACHANKARD2.txt")
dungdaykard_df = pd.read_csv(link+"DUNGDAYKARD.txt")
nghedienthoaikard_df = pd.read_csv(link+"NGHEDIENTHOAIKARD.txt")
ngoixuongkard_df = pd.read_csv(link+"NGOIXUONGKARD.txt")
uongnuockard_df = pd.read_csv(link+"UONGNUOCKARD.txt")
vaytaykard_df = pd.read_csv(link+"VAYTAYKARD.txt")
vaytaykard2_df = pd.read_csv(link+"VAYTAYKARD2.txt")
dibokard_df = pd.read_csv(link+"DIBOKARD.txt")
uongnuocb_df = pd.read_csv(link+"UONGNUOCB.txt")
daubungt_df = pd.read_csv(link+"DAUBUNGT.txt")
daudaut_df = pd.read_csv(link+"DAUDAUT.txt")
daubung2_df = pd.read_csv(link+"DAUBUNG2.txt")

X = []
y = []
no_of_timesteps = 10

dataset = vaytay_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

dataset = votay_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)
dataset = lacnguoi_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)
dataset = daubung_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(3)
dataset = nhucdau_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(4)
dataset = dibo_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(5)
dataset = texiu_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(6)
dataset = daulung_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(7)
dataset = dauchan_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(8)
dataset = dautay_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(9)
dataset = dauco_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(10)
dataset = uongnuoc_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(11)
dataset = texiu2_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(12)
dataset = votay2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(13)

dataset = dachankard_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(14)


dataset = dachankard2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(15)


dataset = dungdaykard_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(16)


dataset = nghedienthoaikard_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(17)


dataset = ngoixuongkard_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(18)


dataset = uongnuockard_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(19)


dataset = vaytaykard_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(20)
dataset = vaytaykard2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(21)
dataset = dibokard_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(22)
dataset = uongnuocb_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(23)
dataset = daubungt_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(24)
dataset = daudaut_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(25)
dataset = daubung2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(26)


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
model.add(Dense(units= 27, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")
#categorical_crossentropy >2 class
#binary_crossentropy =2 class
model.summary()

model.fit(X_train, y_train, epochs=20, batch_size=128,validation_data=(X_test, y_test))
model.save("model.h5")
model.summary()


