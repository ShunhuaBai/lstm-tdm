# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:39:38 2020

@author: ShaunBai
"""


import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# read in raw data
df = pd.read_csv(r'F:\UTA\Research\personal_mobility\TRB_sub\intermediate\mod_in.csv', engine='python')
#df = df.sort_values(by=['date'])

# fix random seed for reproducibility
np.random.seed(7)

def create_dataset(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    d = pd.DataFrame(data)
    cols, names = list(), list()
    #input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(d.shift(-i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(d.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # assemble
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# data preparation
df_mod = df[['ntrips', 'avg_temp', 'prec', 'Type']]

values = df_mod.values
encoder = LabelEncoder()
values[:, 3] = encoder.fit_transform(values[:, 3])
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify number of lag days(6) and number of features(4)
n_days = 6
n_features = 4
n_obs = n_days * n_features 
reframed = create_dataset(scaled, n_days, 1)
values = reframed.values
# Jan-Aug as train set(first 1458 rows (243 days * 6 types), Sep-Dec as test set
train = values[:1458, :]
test = values[1458: , :]
train_X, train_y = train[:, :n_obs], train[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
test_X, test_y = test[:, :n_obs], test[:, -n_features]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# computational time marker
start_time = time.clock()
# fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_X, train_y, epochs=100, batch_size=6, 
          validation_data=(test_X, test_y), verbose=2, shuffle=False)
# elapsed time
time_elapsed = (time.clock() - start_time)
print('Elapsed time: %.2f seconds' % (time_elapsed))
# plot history
plt.plot(history.history['loss'], label='train', color='blue')
plt.plot(history.history['val_loss'], label='validation', color='red')
plt.xlabel('epochs',fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.ylim(0, 0.04)
plt.yticks(ticks=[0, 0.01, 0.02, 0.03, 0.04])
plt.legend()
plt.show()

# make predictions on test set
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
# invert predictions
inv_yhat = np.concatenate((yhat, test_X[:, -3: ]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -3:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test Score: %.2f RMSE' % (testScore))

# make predictions on train set
trainPredict = model.predict(train_X)
train_X = train_X.reshape(train_X.shape[0], n_days*n_features)
# invert training predictions
inv_trainPredict = np.concatenate((trainPredict, train_X[:, -3: ]), axis=1)
inv_trainPredict = scaler.inverse_transform(inv_trainPredict)
inv_trainPredict = inv_trainPredict[:, 0]
train_y = train_y.reshape(len(train_y), 1)
inv_trainY = np.concatenate((train_y, train_X[:, -3: ]), axis=1)
inv_trainY = scaler.inverse_transform(inv_trainY)
inv_trainY = inv_trainY[:, 0]
trainScore = math.sqrt(mean_squared_error(inv_trainY, inv_trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))

destination_types = ['downtown', 'university', 'airport',
                     'inner POI', 'suburban POI', 'fringe POI']



# merge predictions and observed of training and testing data together
merged = df_mod[['ntrips', 'Type']]
merged.rename(columns={'ntrips': 'observed'}, inplace=True)
predicted = np.append(inv_trainPredict, inv_yhat)
predicted = np.append(predicted, [0., 0., 0., 0., 0., 0.])
merged['predicted'] = predicted
merged['date'] = df['date']
merged['weekend'] = df['weekend']
merged['holiday'] = df['holiday']
merged['school'] = df['school']
#calculate squared errors and plot
last_train_date = '2019-08-31'
merged['er'] = merged['predicted'].values - merged['observed'].values
merged['stage'] = 'training'
merged.loc[merged['date']>last_train_date, 'stage'] = 'testing'
# drop last 6 rows that weren't predicted due to back propogation rules
merged.drop(merged.tail(6).index, inplace=True)
merged_train = merged.loc[merged['date']<=last_train_date]
merged_test = merged.loc[merged['date']>last_train_date]
merged.to_csv(r'F:\UTA\Research\personal_mobility\TRB_sub\intermediate\results.csv', index=False)

#outlier properties
flierprops = dict(marker='o', markersize=2, linestyle='none')
# weekend on training
train_weekend_ax = sns.boxplot(data=merged_train, x='er', y='Type', hue='weekend',
                       order=destination_types, palette='Set1', 
                       flierprops=flierprops, showfliers=False, linewidth=1)
train_weekend_ax.set(xlabel='error', ylabel='destination type')
# holiday on training
train_holiday_ax = sns.boxplot(data=merged_train, x='er', y='Type', hue='holiday',
                       order=destination_types, palette='Set1', 
                       flierprops=flierprops, showfliers=False, linewidth=1)
train_holiday_ax.set(xlabel='error', ylabel='destination type')
# school on training
train_school_ax = sns.boxplot(data=merged_train, x='er', y='Type', hue='school',
                       order=destination_types, palette='Set1', 
                       flierprops=flierprops, showfliers=False, linewidth=1)
train_school_ax.set(xlabel='error', ylabel='destination type')

# weekend on testing
test_weekend_ax = sns.boxplot(data=merged_test, x='er', y='Type', hue='weekend',
                       order=destination_types, palette='Set1', 
                       flierprops=flierprops, showfliers=False, linewidth=1)
test_weekend_ax.set(xlabel='error', ylabel='destination type')

# holiday on testing
test_holiday_ax = sns.boxplot(data=merged_test, x='er', y='Type', hue='holiday',
                       order=destination_types, palette='Set1', 
                       flierprops=flierprops, showfliers=False, linewidth=1)
test_holiday_ax.set(xlabel='error', ylabel='destination type')

# school on testing
test_school_ax = sns.boxplot(data=merged_test, x='er', y='Type', hue='school',
                       order=destination_types, palette='Set1', 
                       flierprops=flierprops, showfliers=False, linewidth=1)
test_school_ax.set(xlabel='error', ylabel='destination type')

# plot baseline and predictions
rmse = pd.DataFrame()
rmse['types'] = destination_types
rmse['training_rmse'] = np.nan
rmse['test_rmse'] = np.nan
sns.set(style='white')
fig, axes = plt.subplots(6, 1, figsize=(30,15), sharex=True, sharey=True)
fig.text(0.5, 0.01, 'date', va='center', ha='center', fontsize=20)
fig.text(0.08, 0.5, 'number of trips', va='center', rotation='vertical', fontsize=20)

for i in range(6):
    merged_by_destination = merged[merged['Type']==destination_types[i]]
    merged_by_destination.reset_index(drop=True, inplace=True)
    merged_by_destination_train = merged_by_destination.loc[merged_by_destination['date']<=last_train_date]
    merged_by_destination_test = merged_by_destination[merged_by_destination['date']>last_train_date]
    #calculate RMSE for training and testing by each destination
    rmse_train = math.sqrt(mean_squared_error(merged_by_destination_train['observed'], 
                                              merged_by_destination_train['predicted']))
    rmse_test = math.sqrt(mean_squared_error(merged_by_destination_test['observed'], 
                                              merged_by_destination_test['predicted']))
    rmse.loc[rmse['types']==destination_types[i], 'training_rmse'] = rmse_train
    rmse.loc[rmse['types']==destination_types[i], 'test_rmse'] = rmse_test
    melt = pd.melt(merged_by_destination, id_vars=['date'], 
                   value_vars=['observed', 'predicted'],
                   var_name='cat', value_name='ntrips').sort_values(['date', 'cat'])
    melt.set_index('date', inplace=True)
    xlabels = np.array(melt.index)
    sns.lineplot(data=melt, x=melt.index, y='ntrips', hue='cat', 
                 palette={'observed': 'blue', 'predicted': 'red'}, 
                 ax=axes[i])
    axes[i].set_title(destination_types[i], x=0.05, y=0.75, fontsize=20)
    axes[i].lines[0].set_linestyle('-')
    axes[i].lines[1].set_linestyle('--')
    
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(loc='upper right', fontsize=18, handles=handles[1:], labels=labels[1:])
    axes[i].set(xticks=melt.index, xticklabels=xlabels)
    axes[i].set_ylim(0, 3500)
    axes[i].set_yticks((0, 1000, 2000, 3000))
    axes[i].set_xlabel(None)
    axes[i].set_ylabel(None)
    axes[i].tick_params(labelsize=20)
    for index, label in enumerate(axes[i].xaxis.get_ticklabels()):
        l = label.get_text()
        if l[-2:] != '01':
            label.set_visible(False)
    
plt.xticks(rotation=45)
plt.setp(axes)
plt.savefig(r'F:\UTA\Research\personal_mobility\TRB_sub\Graphs\results.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close('all')

rmse = rmse.append({'types': 'overall', 'training_rmse': trainScore, 'test_rmse': testScore},
            ignore_index=True)
rmse.to_csv(r'F:\UTA\Research\personal_mobility\TRB_sub\intermediate\rmse.csv', index=False)
    
    

