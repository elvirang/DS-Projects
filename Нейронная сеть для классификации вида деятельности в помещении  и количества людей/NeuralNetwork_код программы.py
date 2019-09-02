#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 00:21:35 2019

@author: elviraneganova
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
import plotly
import plotly.graph_objs as go
from keras.utils import to_categorical
filepath = "/Users/elviraneganova/Google Drive/ИАУП 18/Проект нейронная сеть/"
#%%
def plotly_df(df, path, title, names) :
    """Визуализация интерактивного графика"""
    data = []
    for column, i in zip(df.columns, range(0, len(names))):
        trace = go.Scatter(
            x = df.index,
            y = df[column],
            mode = 'lines',
            name = names[i]
            )
        data.append(trace)
    layout = dict(title = title)
    fig = dict(data = data, layout = layout)
    plotly.offline.plot(fig, filename=path+title +'.html', auto_open=False) #сохраняет интерактивный график в html файл в папке с кодом
#%%
#Импортируем данные
adress = filepath + "Данные/datasets-location_A/room_climate-location_A-measurement"
data = []
for i in range (1, 61): #60 наборов данных
    if len(str(i)) == 1 :
        num = "0"+str(i)
    else : 
       num = str(i)
    num = num + ".csv"
    data.append(genfromtxt((adress+num),delimiter=','))

data_train_test =np.vstack((data[0], data[1]))
for i in range(2, len(data)) :
    data_train_test = np.vstack((data_train_test, data[i]))
print("Размерность всех данных:", data_train_test.shape)

#с датафреймом удобнее работать
dataframe = pd.DataFrame(data_train_test, columns = ["id", "time_ms", "time_s", "node", "temp", "hum", "light1", "light2", "occup", "act", "door", "win"]).drop(columns = ["id"])
dataframe.shape
#%%
#X - независимые переменные, Y - зависимые переменные
Y = dataframe.copy()[["occup","act"]]
X = dataframe.copy().drop(columns = ["occup","act"])
#%%
#Визуальный анализ данных
plotly_df(df = X[['temp']].iloc[:21000], path = filepath + "Данные/Графики/", title = 'Температура в комнате А', names = ['temp'])
#%%
from sklearn.preprocessing import MinMaxScaler
#Нормализует данные, приводя их к одной шкале (feature_range) 
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
X_scaled.shape
#%%
#ОПРЕДЕЛЯЕМ КОЛИЧЕСТВО ЧЕЛОВЕК В КОМНАТЕ
#Делим ровно выборки
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y["occup"], test_size=0.33, random_state=42, stratify = Y)
#Параметр stratify сохраняет пропорции зависимой переменной в тестовой и
#обучающей выборках такими же, как и в исходной выборке
#%%
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([
  Dense(64, activation='relu', input_shape=(9,)),
  Dense(64, activation='relu'),
  Dense(3, activation='softmax'),
])
model.summary()

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

history = model.fit(
  np.array(X_train),
  to_categorical(Y_train),
  epochs=40,
  batch_size=32,
  validation_data=(np.array(X_test), to_categorical(Y_test))
)

# Проверка точности предсказания нейронной сети
_, score_test = model.evaluate(np.array(X_test), to_categorical(Y_test), verbose=1) 
_, score_train = model.evaluate(np.array(X_train), to_categorical(Y_train), verbose=1) 
print('Train: %.3f, Test: %.3f' % (score_train, score_test)) 
#%%
#построение графика размера ошибки модели в процессе обучения
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
#построение графика точности модели в процессе обучения
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()
#%%
stats = pd.DataFrame(history.history)
plotly_df(df = stats[["loss", "val_loss"]], path = filepath + "Данные/Графики/",  title = "Сокращение ошибки в процессе обучения нейросети. Определение кол-ва людей в комнате", names = ['train', 'test'])
plotly_df(df = stats[["acc", "val_acc"]], path = filepath + "Данные/Графики/", title = "Повышение точности в процессе обучения нейросети. Определение кол-ва людей в комнате", names = ['train', 'test'])
#%%
#Сохранение весов модели
model.save_weights('model_occupancy.h5')
#%%
#Загрузка модели
#model = Sequential([
#  Dense(64, activation='relu', input_shape=(8,)),
#  Dense(64, activation='relu'),
#  Dense(3, activation='softmax'),
#])
#Загрузка сохраненных ранее весов
#model.load_weights('model.h5')

#Предсказание класса для  пяти строк тестовой выборки
predictions = model.predict(np.array(X_test)[2:7])
print(np.argmax(predictions, axis=1)) #[2 2 0 2 0]
#Сравнение с истинными значениями
print(np.array(Y_test[:5])) #[2. 1. 2. 2. 0.]
#%%
#Сохранение графа нейронной сети
from keras.utils import plot_model
plot_model(model, to_file= filepath + 'Данные/Графики/model1.png', show_shapes=True)
#%%
#ОПРЕДЕЛЯЕМ ВИД ДЕЯТЕЛЬНОСТИ 
#Делим ровно выборки
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y["act"], test_size=0.33, random_state=42, stratify = Y)
#%%
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([
  Dense(64, activation='relu', input_shape=(9,)),
  Dense(64, activation='relu'),
  Dense(5, activation='softmax'),
])
model.summary()

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

from keras.utils import to_categorical
history = model.fit(
  np.array(X_train),
  to_categorical(Y_train),
  epochs=40,
  batch_size=32,
  validation_data=(np.array(X_test), to_categorical(Y_test))
)
#Определение точности прогноза
_, score_test = model.evaluate(np.array(X_test), to_categorical(Y_test), verbose=1) 
_, score_train = model.evaluate(np.array(X_train), to_categorical(Y_train), verbose=1) 
print('Train: %.3f, Test: %.3f' % (score_train, score_test)) 
#%%
stats = pd.DataFrame(history.history)
#построение графика размера ошибки модели в процессе обучения
plt.subplot(211)
plt.title('Loss')
plt.plot(stats['loss'], label='train')
plt.plot(stats['val_loss'], label='test')
plt.legend()
#построение графика точности модели в процессе обучения
plt.subplot(212)
plt.title('Accuracy')
plt.plot(stats['acc'], label='train')
plt.plot(stats['val_acc'], label='test')
plt.legend()
plt.show()
#%%
stats = pd.DataFrame(history.history)
plotly_df(df = stats[["loss", "val_loss"]], path = filepath + "Данные/Графики/", title = "Сокращение ошибки в процессе обучения нейросети. Определение вида деятельности", names = ["train", "test"])
plotly_df(df = stats[["acc", "val_acc"]], path = filepath + "Данные/Графики/", title = "Повышение точности в процессе обучения нейросети. Определение вида деятельности", names = ['train', 'test'])
#%%
#Сохранение весов модели
model.save_weights('model_activity.h5')
#%%
#Загрузка модели
#model = Sequential([
#  Dense(64, activation='relu', input_shape=(8,)),
#  Dense(64, activation='relu'),
#  Dense(3, activation='softmax'),
#])
#Загрузка сохраненных ранее весов
#model.load_weights('model.h5')

#Предсказание класса для первых пяти строк тестовой выборки
predictions = model.predict(np.array(X_test)[:5])
print(np.argmax(predictions, axis=1)) #[4 2 1 2 0]
#Сравнение с истинными значениями
print(np.array(Y_test[:5])) #[4. 2. 1. 2. 0.]
#%%
#Отрисовка архитектуры
plot_model(model, to_file= filepath + 'Данные/Графики/model2.png', show_shapes=True)
#%%