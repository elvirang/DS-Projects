# -*- coding: utf-8 -*-

"""
    Построение прогноза по временному ряду потребления энергоресурсов
"""
#%% Предобработка данных
import pandas as pd
import numpy as np

def get_files():
    """Формирование списка используемых файлов.
Возвращает сформированный список и название колонок для будущего набора данных"""
    list_of_files = []
    list_of_files.append('lifts_6-7.xls')
    list_of_files.append('light_1-2.xls')
    list_of_files.append('light_6-7.xls')
    list_of_files.append('ouk_1-2.xls')
    list_of_files.append('ouk_6-7.xls')
    list_of_value_columns = ['lifts_6_7', 'light_1_2', 'light_6_7', 'ouk_1_2', 'ouk_6_7']
    return list_of_files, list_of_value_columns

def read_excel(file_name):
    """Читает файлы и возвращает список и прочтённых данных без обработки (сырые данные)"""
    return np.array(pd.read_excel(file_name))

def transform_data(data_from_file):
    """Преобразуем данные в "удобный" формат.
Вытаскиваем дату из ячейки, затем добавляем эту дату к какждому временному промежутку. Каждое значение измеряемого параметра в каждые 30 минут записываем в отдельный список numbers.
озращаем список с датой-временем и список с измеренными значениями."""
    dates=[]
    numbers=[]
    date=''
    for lines in data_from_file:
        if not(pd.isnull(lines[0])):
            date = lines[0][8:18]
        else:
            if not(pd.isnull(lines[1])):
                date += ' ' + lines[1]
            dates.append(date)
            date = date[0:10]
        if not(pd.isnull(lines[-1])):
            numbers.append(float(lines[-1]))
    return dates, numbers

def to_list(tuple_of_lists):
    """Возвращаем список из неизменяемого списка, для дальнейшей обработки."""
    temp=[]
    transform_data = []
    for i, lines in enumerate(tuple_of_lists[0]):
        for col in tuple_of_lists:
            temp.append(col[i])
        transform_data.append(temp)
        temp=[]
    return transform_data

def to_timeseries_df(_list, name_value):
    """Приведение данных к данным временного ряда. Определяем формат индекса даты-времени и устанавливаем его для набора данных.
Возвращаем датафрейм с индексом типа дэйттайм."""
    timeformat = '%d.%m.%Y %H:%M:%S'
    df = pd.DataFrame(_list, columns=['Date', name_value])
    df.Date = pd.to_datetime(df.Date, format=timeformat)
    df = df.set_index('Date')
    return df

def join_parts(files):
    """Объединение всех датафреймов, сформированным по данным из файлов в один датафрейм по индексу времени.
Возвращаем объединённый датафрейм"""
    dfs = []
    for i, file in enumerate(files[0]):
        temp_df = to_timeseries_df(to_list(transform_data(read_excel(file))), files[1][i])
        dfs.append(temp_df)
    return pd.concat(dfs, axis = 1, join_axes=[temp_df.index])

dataframe = join_parts(get_files())
print(dataframe)

print("Проверка на пропуск данных\n", dataframe.isnull().sum()) #Нули означают, что пропущенных данных нет, иначе - кол-во пропущенных данных

def null_values(df) :
    """Подсчет кол-ва пропущенных данных"""
    n = [] #кол-во пропущенных значений в каждом столбце
    col = [] #названия столбцов, в которых пропущены значения
    n = df.isnull().sum() #считаем кол-во пропущенных значений в каждом столбце
    for i in range(len(df.columns)):
        if n[i] != 0 : #если нулевых значений нет, сумма будет 0. Иначе - показываем, сколько пропусков в столбце
            print(n[i], "empty values are in column", df.columns[i])
            col.append(df.columns[i]) #сохраняем названия столбцов с пропусками
    if len(col) == 0 :
        print("No null values were found")
    return col #возвращаем названия столбцов с пропусками, чтобы в дальнейшем можно было их применить в функции, заполняющей пропуски

null_columns = null_values(dataframe) #Находим пропуски в данных

null_columns
#Если бы нужно было заполнить данные, например:
#dataframe= dataframe.fillna(method='bfill')

# проверка типов данных и пустот
dataframe.info()
# Начальный вид данных из экселя
df_ex = pd.read_excel(get_files()[0][0])

#%% Визуализация данных
from plotly import __version__
from plotly.offline import download_plotlyjs, plot, iplot
from plotly import graph_objs as go

def plotly_df(df, title):
    """Визуализация интерактивного графика"""
    data = []
    
    for column in df.columns:
        trace = go.Scatter(
            x = df.index,
            y = df[column],
            mode = 'lines',
            name = column
        )
        data.append(trace)
    
    layout = dict(title = title)
    fig = dict(data = data, layout = layout)
    plot(fig, filename=title, auto_open=False) #сохраняет интерактивный график в html файл в папке с кодом


plotly_df(dataframe[dataframe.columns.tolist()[-2:]], title = "Потребление электроэнергии по квартирам") #сохраняет интерактивный график в html файл в папке с кодом

plotly_df(dataframe[dataframe.columns.tolist()[0:-2]], title = "Использование энергии лифтами и на освещение в подъездах") #сохраняет интерактивный график в html файл в папке с кодом

#%% Получение статистических данных
dataframe.describe()#статистики count, mean, std, min, 25%-50%-75% percentile, max

#Сумма энергий за месяц 
sum_elev = dataframe['lifts_6_7'].sum()
sum_light1 = dataframe['light_1_2'].sum()
sum_light2 = dataframe['light_6_7'].sum()
sum_apart1 = dataframe['ouk_1_2'].sum()
sum_apart2 = dataframe['ouk_6_7'].sum()
print("Сумма электроэнергии за месяц, потраченная на лифт во 2 подъезде: %(sum_elev)0.3f,\n"
      "Сумма электроэнергии за месяц, потраченная на освещение в 1 подъезде: %(sum_light1)0.3f,\n"
      "Сумма электроэнергии за месяц, потраченная на освещение в 2 подъезде: %(sum_light2)0.3f,\n"
      "Сумма электроэнергии за месяц, потраченная в квартирах в 1 подъезде: %(sum_apart1)0.3f,\n"
      "Сумма электроэнергии за месяц, потраченная в квартирах в 2 подъезде: %(sum_apart2)0.3f,\n"
      %{"sum_elev": float(sum_elev), "sum_light1": float(sum_light1), "sum_light2": float(sum_light2), "sum_apart1": float(sum_apart1), "sum_apart2": float(sum_apart2)})

#%% ПРОГНОЗ
import matplotlib as mp
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime as dt

from scipy import stats
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (15, 18)

#Определяем ряд, с которым работаем
column = 'lifts_6_7'
columns_todrop = [x  for x in dataframe.columns if x != column]
columns_todrop
temp = dataframe.copy().drop(columns = columns_todrop)

plt.rcParams['figure.figsize'] = (15, 8) 
ax = sm.tsa.seasonal_decompose(temp, freq = 48).plot() #за сутки 48 наблюдений
plt.show()

#Проверяем стационарность ряда
adf = sm.tsa.stattools.adfuller(temp[column])
print(  "Тестовая статистика:", round(adf[0],3),
        "\nКритические значения: ", adf[4],
        "\np-value:", round(adf[1], 0))
#Критерий Dickey-Fuller отвергает нулевую гипотезу, но мы все еще видим тренд
#Проведем дифференцирование

temp[column+'_diff'] = temp[column] - temp[column].shift(1)
sm.tsa.seasonal_decompose(temp[column+'_diff'][1:], freq = 48).plot()
adf = sm.tsa.stattools.adfuller(temp[column+'_diff'][1:])
print(  "Тестовая статистика:", round(adf[0],3),
        "\nКритические значения: ", adf[4],
        "\np-value:", round(adf[1], 0))
#Критерий Dickey-Fuller отвергает нулевую гипотезу, тренда нет
#Ряд стационарен

#ax =  plt.subplot(211)
#sm.graphics.tsa.plot_acf(temp[column][1:].values.squeeze(), 
#                         lags = 50, ax=ax)
sm.graphics.tsa.plot_acf(temp[column][1:].values.squeeze(), 
                         lags = 50)
plt.show()

sm.graphics.tsa.plot_pacf(temp[column][1:].values.squeeze(), 
                          lags=50)
plt.show()

d = 1 #дифференцировали один раз
q = 1 #первый лаг имеет значительную автокорелляцию по ACF
D = 1 #наблюдается сезонность
S = 48#самое большое значение на ACF на 48 лаге
p = 1 #первый лаг имеет значительную автокорелляцию по PACF
P = 1 #ACF положительно на 48 лаге
Q = 0 #ACF положительно на 48 лаге

#определяем границы, в которых перебираем параметры
ps = range(0, 2)
ds = range(0, 2)
qs = range(0, 2) 
Ps = range(0, 2)
Ds = range(0, 2)
Qs = range(0, 2)

#ищем все возможные комбинации параметров с помощью метода product
from itertools import product
parameters = product(ps, ds, qs, Ps, Ds, Qs)
parameters_list = list(parameters)
print(len(parameters_list))

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

results = []
best_aic = float("inf")

warnings.filterwarnings('ignore')

for param in tqdm(parameters_list):
    # некоторые комбинации приводят к ошибкам вычислений, поэтому помещаем в структуру try-except
    try:
                                #SARIMA(p,d,q)(P,D,Q)[S]
        model=sm.tsa.statespace.SARIMAX(temp[column], order=(param[0], param[1], param[2]), 
                                        seasonal_order=(param[3], param[4], param[5], 48)).fit(disp=-1) 
                                        #disp- Сейчас установлено False.
                                        #      Если установить True, то будут выводиться сообщения о сходимости.
    except ValueError:
        print('неверные параметры:', param)
        continue
    aic = model.aic # для проверки качесвта модели расчитывается критерий Акаике
# Если критерий Акаике оказывается меньше, чем в текущей лучшей модели, то сохраняем новую модель в best model, её оценку и лучшие параметры.
    if aic < best_aic: #Akaike Information Criterion
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
    
warnings.filterwarnings('default')

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head()) #берем только верхние 5 лучших

print(best_model.summary()) #выводим статистику по лучшей модели

# Строим график остатков модели
plotly_df(pd.DataFrame(best_model.resid[1:]), "Остатки") #сохраняет интерактивный график в html файл в папке с кодом

#Выводим график автокорреляции остатков

sm.graphics.tsa.plot_acf(best_model.resid[1:].values.squeeze(), lags=48)
plt.show()

print("Критерий Стьюдента: p=%f" % stats.ttest_1samp(best_model.resid[1:], 0)[1]) #проверка соответствия матожидания 0
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[1:])[1]) #проверка стационарности

#накладываем на график прогнозные значения на реальные
plt.figure(figsize=(18, 10))
temp['model'] = best_model.fittedvalues
plotly_df(temp[[column, 'model']], "Прогнозные значения и реальные значения") #сохраняет интерактивный график в html файл в папке с кодом

#Измерим качество получившейся модели
#Вручную:
temp['e'] = abs(temp[column] - temp['model'])
temp['p'] = 100*temp['e']/temp[column]
print('MAPE', np.mean(abs(temp['p'])))
print('MAE', np.mean(abs(temp['e'])))
#В результате получили:
# MAPE (средняя абсолютная ошибка модели в процентах) = 22
# MAE (средняя абсолютная ошибка модели) = 0.07

#R квадрат - качество модели
R_square=(1-((temp[column] - best_model.fittedvalues)**2).sum()/((temp[column] - temp[column].mean())**2).sum())
print(R_square)

"""Прогнозирование с помощью модели SARIMAX"""
from dateutil.relativedelta import relativedelta

columns_todrop = [x  for x in temp.columns if x != column]
columns_todrop
data = temp.copy().drop(columns = columns_todrop)
print(data.shape)

#хотим предсказать неделю начиная с 2015-02-01 00:00:00
date_list = [pd.datetime.strptime("2015-02-01 00:00:00", "%Y-%m-%d %H:%M:%S") + 
             # relativedelta - создание новых значений на указанном промежутке
             # 336 / 48 = 7 дней прогноза
             relativedelta(hours=0.5*x) for x in range(0,336)] 
future = pd.DataFrame(index=date_list, columns=data.columns)
# задаём границу, по которым будем разбивать временной ряд для отрисовки графика
future_threshold = 336

df = pd.DataFrame(data=list(best_model.predict(start = 1488, end = 1823)),
                  index=date_list)

# добавляем к реальным значениям прогнозные
df_forecast = data.append(df.rename({0:column}, axis='columns')[1:])

plt.rcParams['figure.figsize'] = (20, 5)
# разбор массива на промежутки реальных значений и прогноза (для выделения на графике разными цветами)
df_forecast[column][:-future_threshold].plot(color='b', label='actual'); #все строки кроме последних 31
df_forecast[column][-future_threshold-1:].plot(color='red', label='forecast'); #начиная с 32 с конца
plt.ylabel('Data');
plt.legend();

plt.show()
