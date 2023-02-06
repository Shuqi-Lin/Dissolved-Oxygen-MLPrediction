# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:23:06 2022

@author: Shuqi Lin
"""
#%% Import packages
import os
import pandas as pd
import numpy as np
import datetime
import random
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pickle
pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 1000)  # or 1000
import time
from numpy import concatenate
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

#%%
# Modify iceoff day factor into category
def modify_iceoff_day(x,iceoff_name):
    if x[iceoff_name]>=30:
        return 3
    elif (x[iceoff_name]<30)&(x[iceoff_name]>=20):
        return 2
    elif (x[iceoff_name]<20)&(x[iceoff_name]>=10):
        return 1
    elif (x[iceoff_name]<10)&(x[iceoff_name]>=-10):
        return 0
    elif (x[iceoff_name]<-10)&(x[iceoff_name]>=-20):
        return -1
    elif (x[iceoff_name]<-20)&(x[iceoff_name]>=-30):
        return -2
    else:
        return -3
## Modifiy ice duration
def modify_ice_d(x,ice_d_name):
    if x[ice_d_name]>60:
        return 'long'
    elif (x[ice_d_name]<=60) & (x[ice_d_name]>30):
        return 'middle'
    elif (x[ice_d_name]<=30) & (x[ice_d_name]>0):
        return 'short'
    else:
        return 'no_ice_d'

# Modify zooplankton 
def modify_Daphnia(x,Daphnia):
    if x[Daphnia]>400:
        return 'high'
    elif (x[Daphnia]<=400) & (x[Daphnia]>200):
        return 'middle'
    elif (x[Daphnia]<=200) & (x[Daphnia]>50):
        return 'low'
    else:
        return 'little'

# Load testing dataset(Daily data)
def read_daily_test_df(features,Ice_c,iceoff_name,ice_d_name,zoo_name,file): # test_time = pd.Timestamp(yyyy,mm,dd)
    Lake_HydMet = pd.read_csv(file,header = 0,sep = '\t',parse_dates = ['Date'])
    #Ice categorization
    if Ice_c=='Y':
        Lake_HydMet[iceoff_name]=Lake_HydMet.apply(lambda x:modify_iceoff_day(x,iceoff_name), axis=1)
        Lake_HydMet[iceoff_name]=pd.Categorical(Lake_HydMet[iceoff_name])
        Lake_HydMet[ice_d_name]=Lake_HydMet.apply(lambda x:modify_ice_d(x,ice_d_name), axis=1)
        Lake_HydMet[ice_d_name]=pd.Categorical(Lake_HydMet[ice_d_name])
        Lake_HydMet=pd.concat([Lake_HydMet,pd.get_dummies(Lake_HydMet[[iceoff_name,ice_d_name]])],
                      axis=1).drop([iceoff_name,ice_d_name],axis=1)              
    if zoo_c=='Y':
        Lake_HydMet[zoo_name]=Lake_HydMet.apply(lambda x:modify_Daphnia(x,zoo_name), axis=1)
        Lake_HydMet[zoo_name]=pd.Categorical(Lake_HydMet[zoo_name])
        Lake_HydMet=pd.concat([Lake_HydMet,pd.get_dummies(Lake_HydMet[[zoo_name]])],
              axis=1).drop([zoo_name],axis=1)              

    Lake_HydMet = Lake_HydMet[features]
    #Lake_HydMet = Lake_HydMet[(Lake_HydMet['Date']>=test_time[0])&(Lake_HydMet['Date']<=test_time[1])].dropna()
    return Lake_HydMet

# convert series to supervised learning
def series_to_supervised(data, n_in, n_out, var_name,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1] # number of variables
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(var_name[j]+'(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(var_name[j]+'(t)') for j in range(n_vars)]
        else:
            names += [(var_name[j]+'(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg

# Load testing dataset(Daily data)
def read_daily_test_df(features,Ice_c,iceoff_name,ice_d_name,zoo_name,file): # test_time = pd.Timestamp(yyyy,mm,dd)
    Lake_HydMet = pd.read_csv(file,header = 0,sep = '\t',parse_dates = ['Date'])
    #Ice categorization
    if Ice_c=='Y':
        Lake_HydMet[iceoff_name]=Lake_HydMet.apply(lambda x:modify_iceoff_day(x,iceoff_name), axis=1)
        Lake_HydMet[iceoff_name]=pd.Categorical(Lake_HydMet[iceoff_name])
        Lake_HydMet[ice_d_name]=Lake_HydMet.apply(lambda x:modify_ice_d(x,ice_d_name), axis=1)
        Lake_HydMet[ice_d_name]=pd.Categorical(Lake_HydMet[ice_d_name])
        Lake_HydMet=pd.concat([Lake_HydMet,pd.get_dummies(Lake_HydMet[[iceoff_name,ice_d_name]])],
                      axis=1).drop([iceoff_name,ice_d_name],axis=1)              

    if zoo_c=='Y':
        Lake_HydMet[zoo_name]=Lake_HydMet.apply(lambda x:modify_Daphnia(x,zoo_name), axis=1)
        Lake_HydMet[zoo_name]=pd.Categorical(Lake_HydMet[zoo_name])
        Lake_HydMet=pd.concat([Lake_HydMet,pd.get_dummies(Lake_HydMet[[zoo_name]])],
                              axis=1).drop([zoo_name],axis=1)              
    Lake_HydMet = Lake_HydMet[features]
    #Lake_HydMet = Lake_HydMet[(Lake_HydMet['Date']>=test_time[0])&(Lake_HydMet['Date']<=test_time[1])].dropna()
    return Lake_HydMet

# reframe dataset
def reframe(values,hyperparameters,var_names):
    reframed = series_to_supervised(values, hyperparameters['time_steps'], hyperparameters['n_out'],var_names)
    reframed = reframed.iloc[hyperparameters['time_steps']:]
    drop_col =[]
    n_var = len(var_names)
    for i in range(1,hyperparameters['time_steps']+1):
        drop_col += [n_var*i-1]
    reframed.drop(reframed.iloc[:,drop_col],axis=1,inplace = True)
    return reframed

# load dataset
def load_dataset(df,var_name):
    values = df[var_name].values
    return values

def sparse_dataset(data_X,data_y):
    index = []
    y = []
    for i in range(len(data_y)):
        if ~np.isnan(data_y[i]):
            index.append(i)
            y.append(data_y[i])
    X = np.stack(data_X[index,:,:])
    y = np.array(y)
    return index,X,y

def fit_lstm(train_X,train_y,n_batch,nb_epoch,n_neuros,dropout,verbose,loss_function):
    # design network
    model = Sequential()
    model.add(LSTM(n_neuros,  return_sequences = True,
              input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(n_neuros, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(n_neuros))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss=loss_function, optimizer='adam')
    # fit network
    model.fit(train_X,train_y,epochs =nb_epoch,batch_size = n_batch,verbose = verbose)
    return model

def split_dataset(train,test,time_steps):
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], time_steps+1, int(train_X.shape[1]/(time_steps+1))))
    test_X = test_X.reshape((test_X.shape[0], time_steps+1, int(test_X.shape[1]/(time_steps+1))))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print('number of input timesteps: {}'.format(train_X.shape[1]))
    print('number of features: {}'.format(train_X.shape[2]))
    return train_X, train_y,test_X, test_y

def plot_comparison(df,y,yhat,yhat_ts,n_date,time_steps,index,rmse,mae,r2,nutrient,test_time,ylim):
    # Observation time
    time = df['Date'].iloc[n_date+time_steps:].reset_index()['Date'].iloc[index] 
    # Direct comparison of observation and prediction [yhat] (data point to data point)
    pred = pd.DataFrame(concatenate((yhat.reshape(yhat.shape[0],1),y.reshape(y.shape[0],1)), axis=1),
                        index = time)
    pred.columns = ['Prediction','True value']
    # Extract the continuous timeseries from input dataset
    time_ts = df['Date'].iloc[n_date+time_steps:]
    # The continuous prediction yhat_ts 
    pred_ts = pd.DataFrame(yhat_ts,index = time_ts,columns = ['Prediction'])
    # Compute the 7d rolling mean of the timeseries predction
    pred_ts['Prediction_7d'] = pred_ts['Prediction'].rolling(7,min_periods = 1).mean()
    # Create a continous timeseries without winter gap
    Date = pd.DataFrame(pd.date_range(start = time_ts.iloc[0],
                                  end = time_ts.iloc[-1]),
                    columns = ['Date'])
    pred_ts_gap = Date.merge(pred_ts,how = 'left',on = 'Date')
    f1,ax1 = plt.subplots(1,2,figsize = (18,6),gridspec_kw={'width_ratios': [2, 1]})
    pred_ts_gap.plot(x = 'Date',y = ['Prediction','Prediction_7d'],
                     style = {'Prediction':'b-','Prediction_7d':'k-'},
                     ax = ax1[0])
    pred.plot(y = 'True value',style='ro',alpha = 0.7,ms = 7,ax = ax1[0])
    ax1[0].set_ylabel(nutrient)
    ax1[0].set_xlim((test_time[0],test_time[1]))
    ax1[0].set_ylim(ylim)
    ax1[0].text(0.7, 0.9, 'RMSE:{}, MAE:{}'.format(round(rmse,2),round(mae,2)), 
            horizontalalignment='center',verticalalignment='center', 
            transform=ax1[0].transAxes,fontsize='x-large')
    ax1[0].legend(frameon=False)
    pred.plot(x = 'True value', y = 'Prediction',kind = 'scatter',s = 20,c = 'blue',ax = ax1[1])
    ax1[1].plot(pred['True value'],pred['True value'],lw  =1.5,color = 'black')
    ax1[1].text(0.5, 0.8, 'R2:{}'.format(round(r2,2)), 
                horizontalalignment='center',verticalalignment='center', 
                transform=ax1[1].transAxes,fontsize='x-large')
    return f1,pred_ts

# ensure all data is float
def predict_lstm(df,values,var_name,nutrient,test_time,hyperparameters,ylim):
    n_date = df[(df['Date']<test_time[0])|(df['Date']>test_time[1])]['Date'].count() # number of dates used for trainning
    if df[(df['Date']<test_time[0])].empty:
        train = values[-n_date-1:-1, :] # Testing years are in front of training years: train = values[-n_date-1:-1, :]
        test = values[:-n_date, :] # Testing years are in front of training years: test = values[:-n_date, :]
    if df[(df['Date']>test_time[1])].empty:
        train = values[:n_date, :] # Testing years are after training years: train = values[:n_date, :]
        test = values[n_date:, :] # Testing years are after training years: test = values[n_date:, :]
    train_X, train_y,test_X, test_y = split_dataset(train,test,hyperparameters['time_steps'])
    train_y=Scaler_y.fit_transform(train_y.reshape(-1, 1))
    # fit the lstm model
    index,X,y = sparse_dataset(train_X,train_y) # stack the timeseries input together to create a 2D training input X, and a 1D lable y
    #y_scaled = Scaler.fit_transform(y.reshape(-1,1))
    print('number of samples: {}'.format(len(index)))
    kf = KFold(n_splits=5, shuffle=True, random_state=101) # 5-fold cross validation
    RMSE = []
    MAE = []
    R2 = []
    # fit the lstm model 
    for train_index, test_index in kf.split(X,y):
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = fit_lstm(X_train,y_train,hyperparameters['n_batch'],hyperparameters['nb_epoch'],
                         hyperparameters['n_neuros'],hyperparameters['dropout'],
                         hyperparameters['verbose'],hyperparameters['loss_function'])
        yhat = Scaler_y.inverse_transform(model.predict(X_test,batch_size = hyperparameters['n_batch']))
        y_test = Scaler_y.inverse_transform(y_test.reshape(-1, 1))
        rmse = mean_squared_error(y_test, yhat,squared=False)
        mae= mean_absolute_error(y_test, yhat)
        r2 =  r2_score(y_test, yhat)
        RMSE.append(rmse) 
        MAE.append(mae)
        R2.append(r2)
        #print('Training RMSE: %.2f' %rmse)       
    # make a prediction
    model = fit_lstm(X,y,hyperparameters['n_batch'],hyperparameters['nb_epoch'],
                     hyperparameters['n_neuros'],hyperparameters['dropout'],
                     hyperparameters['verbose'],hyperparameters['loss_function'])
    index,X,y = sparse_dataset(test_X,test_y)#sparse_dataset(test_X,Scaler_y.inverse_transform(test_y.reshape(-1, 1))) # index is the time series
    yhat = Scaler_y.inverse_transform(model.predict(X,batch_size = hyperparameters['n_batch']))
    rmse = mean_squared_error(y, yhat,squared=False)
    mae = mean_absolute_error(y,yhat)
    r2 = r2_score(y, yhat)
    print("Training dataset RMSE %.2f (+/- %.2f)" % (np.mean(RMSE), np.std(RMSE)))
    print("Training dataset MAE %.2f (+/- %.2f)" % (np.mean(MAE), np.std(MAE)))
    print("Training dataset R2 %.2f (+/- %.2f)" % (np.mean(R2), np.std(R2)))
    # make a prediction for the whole timeseries
    yhat_ts = Scaler_y.inverse_transform(model.predict(test_X,batch_size = hyperparameters['n_batch']))
    figure,pred_ts = plot_comparison(df,y,yhat,yhat_ts,n_date,hyperparameters['time_steps'],index,rmse,mae,r2,nutrient,test_time,ylim)
    return model,rmse,mae,r2,figure,pred_ts

def compare(Erken_Nut,Nut_memory,nutrient,Lake_Nut_metrics,hat):
    compare = Erken_Nut.merge(Nut_memory,on = 'Date',how = 'left')[['Date',nutrient+'_x',nutrient+'_y']].dropna()
    compare.columns = [['Date','ML','OB']]
    Lake_Nut_metrics[nutrient].MAE= mean_absolute_error(compare['OB'], compare['ML'])
    Lake_Nut_metrics[nutrient].RMSE = mean_squared_error(compare['OB'], compare['ML'],squared=False)
    Lake_Nut_metrics[nutrient].R2 = r2_score(compare['OB'], compare['ML'])
    # Add the time-series prediction into sample dataset for next variable modeling
    Nut_memory.loc[Nut_memory['year'].isin(test_yr),nutrient] = hat
    return Lake_Nut_metrics,Nut_memory

def predict_ts(df,nutrient,model,hyperparameters,values):
    # add the predictive values into dataset
    value_X, value_y = values[:, :-1], values[:, -1]
    value_X = value_X.reshape((value_X.shape[0], hyperparameters['time_steps']+1, int(value_X.shape[1]/(hyperparameters['time_steps']+1))))
    y_pred = Scaler_y.inverse_transform(model.predict(value_X,batch_size = hyperparameters['n_batch']))    
    df[nutrient].iloc[hyperparameters['time_steps']:]=y_pred[:,0]
    df[nutrient].fillna(method = 'backfill',inplace = True)
    return df

#%%
# Make sure you are in the main folder('..\Dissolved-Oxygen-MLPredictiona')
cd = os.getcwd()
while cd.split('\\')[-1]!='Dissolved-Oxygen-MLPrediction':
    os.chdir('..')
    cd=os.getcwd()
os.chdir(cd+'\\Training dataset')
# Load training dataset
os.chdir(cd+'\\Training dataset')
lakename='Erken'
all_df = pd.read_csv('Erken_Observation_df_nowinter.csv',sep = '\t',parse_dates=['Date'])

#%% Preprocess training features
print(all_df.columns)
Ice_variable='Y' #input('Ice variables? (Y/N)')
if Ice_variable=='Y':
    ## Modifiy iceoff_day variable
    iceoff_name='days from iceoff'# input('name of iceoff variable:')

    modify='Y' #input('Modify iceoff_day? (Y/N)')
    if modify=='Y':
        all_df[iceoff_name]=all_df.apply(lambda x:modify_iceoff_day(x,iceoff_name), axis=1)

    ## categorize ice information
    all_df[['Date',iceoff_name]].plot(x='Date',y=iceoff_name,style='bo',figsize=(18,4))
    all_df[iceoff_name]=pd.Categorical(all_df[iceoff_name])

    ice_d_name='Ice_d'#input('Name of ice duration variable:')
    modify='Y'#input('Modify ice duration? (Y/N)')
    if modify=='Y':
        all_df[ice_d_name]=all_df.apply(lambda x:modify_ice_d(x,ice_d_name), axis=1)

    ## categorize ice information
    all_df[ice_d_name]=pd.Categorical(all_df[ice_d_name])

zoo_variable='N'#input('Zooplankton variable? (Y/N)')
if zoo_variable=='Y':
    zoo_name=input('name of zooplankton: ')
    modify=input('Modify zooplankton level? (Y/N)')
    if modify=='Y':
        all_df[zoo_name]=all_df.apply(lambda x:modify_Daphnia(x,zoo_name), axis=1)
    ## categorize zooplankton
    all_df[zoo_name]=pd.Categorical(all_df[zoo_name])
else:
    zoo_name=''
    
## Get variables dommies
test_df=all_df.copy()
while True:
    binary_f=''#input('Name of binary feature: ')
    if binary_f=='':
        break
    test_df[binary_f]=pd.Categorical(test_df[binary_f])
Ice_c='Y'#input('Ice categorization? (Y/N)')
if Ice_c=='Y':
    test_df=pd.concat([test_df,pd.get_dummies(test_df[[iceoff_name,ice_d_name]])],axis=1).drop([iceoff_name,ice_d_name],axis=1)              
zoo_c='N'#input('Zooplankton categorization? (Y/N)')
if zoo_c=='Y':
        test_df=pd.concat([test_df,pd.get_dummies(test_df[[zoo_name]])],axis=1).drop([zoo_name],axis=1)              

all_df=test_df.copy()
print(all_df.columns)
n_feature = 24
features = ['Date','delT', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow', 'outflow', 
            'thermD', 'MLD', 'W', 'accum_BotT','St',
            'days from iceoff_-3', 'days from iceoff_-2', 'days from iceoff_-1',
            'days from iceoff_0', 'days from iceoff_1', 'days from iceoff_2',
            'days from iceoff_3', 'Ice_d_long', 'Ice_d_middle', 'Ice_d_short']

## Test time
training_yr_df={}
test_time = [pd.Timestamp(2017,1,1),pd.Timestamp(2020,12,31)]
test_yr=list(range(test_time[0].year,test_time[1].year+1))

Nut_num = 4#int(input('How many nutrients to predict? (include Chl and put them in order) '))
Nut=['DO_sur','DO_bot','PO4','TotP']
Nut_f = ['Date']+Nut
Nut_df=all_df[Nut_f]

Lake_Nut_metrics = pd.DataFrame(columns = Nut_f[1:],index=['MAE','RMSE','R2'])
# set the hyperparameters
hyperparameters = {'n_batch':20,'nb_epoch':100,'n_neuros':100,'dropout':0.2,'time_steps':7,
                   'n_out':1,'verbose':0,'loss_function':'mae'}
## Dataset_range
Dataset_range = [pd.Timestamp(all_df['Date'].apply(lambda d:d.year).unique()[0],1,1),
             pd.Timestamp(all_df['Date'].apply(lambda d:d.year).unique()[-1],12,31)]
print('Data spans from {} to {}'.format(Dataset_range[0],Dataset_range[1]))

#%%
## Specify the training year (take 16 years from 1999-2016)
for i in range(30):
    year = all_df['Date'].apply(lambda d:d.year).unique()
    #print('Dataset contains years:{},in total {} years'.format(year,len(year)))
    Dataset_range= [pd.Timestamp(year[0],1,1),
                    pd.Timestamp(year[-1],12,31)]
    year = [j for j in year if j<test_time[0].year]
    random.shuffle(year)
    training_yr=year[0:16]
    training_yr.sort()
    ids=map(id,list(training_yr_df.values()))
    if id(training_yr) not in ids:
        training_yr_df[i]=training_yr
        start_time = time.time()
        os.chdir(cd+'\\Trainning data')
        # Load daily physical factors
        features = ['Date','delT', 'U', 'AirT', 'Humidity', 'CC', 'Prec', 'SWR', 'inflow', 'outflow', 
                    'thermD', 'MLD', 'W', 'accum_BotT',
                    'days from iceoff_-3', 'days from iceoff_-2', 'days from iceoff_-1',
                    'days from iceoff_0', 'days from iceoff_1', 'days from iceoff_2',
                    'days from iceoff_3', 'Ice_d_long', 'Ice_d_middle', 'Ice_d_short']

        file = lakename+'_Daily_Observation_df_nowinter.csv'
        Daily_df = read_daily_test_df(features,Ice_c,iceoff_name,ice_d_name,zoo_name,file).dropna() 
        Daily_df = Daily_df[Daily_df['Date']>Dataset_range[0]]
        os.chdir(cd+'\\Nutrient prediction\\Erken\\direct LSTM')
        features.pop(0) # Pop 'Date'
        Scaler_X = MinMaxScaler()
        Scaler_y = MinMaxScaler()#PowerTransformer(standardize=False)
        start_time = time.time()
        for j in range(Nut_num):
            nutrient=Nut_f[j+1]
            if j==0:
                Nut_memory = Daily_df.merge(pd.concat([Nut_df['Date'],
                                                       Nut_df[nutrient]],
                                                      axis = 1).dropna().drop_duplicates('Date'),
                                            how = 'left',on = 'Date')
            else: 
                Nut_memory = Nut_memory.merge(pd.concat([Nut_df['Date'],
                                                       Nut_df[nutrient]],
                                                      axis = 1).dropna().drop_duplicates('Date'),
                                            how = 'left',on = 'Date')

            print(nutrient)
            features.append(nutrient)
            values = load_dataset(Nut_memory,features)
            # frame as supervised learning
            reframed = reframe(values,hyperparameters,features)
            reframed_scaled=pd.DataFrame(Scaler_X.fit_transform(reframed.iloc[:,:-1]),
                                         columns=reframed.columns[:-1])
            #target_scaled=pd.Series(Scaler_y.fit_transform(reframed.iloc[:,-1].values.reshape(-1, 1)).reshape(-1),
            #                 name=reframed.columns[-1])
            target_array=pd.Series(reframed.iloc[:,-1].values.reshape(-1, 1).reshape(-1),
                         name=reframed.columns[-1])

            reframed_scaled=pd.concat([reframed_scaled,target_array],axis=1)
            values = reframed_scaled.values
            ylim = (0,all_df[nutrient].max())
            start_time = time.time()
            Nut_model,rmse,mae,r2,figure,pred_ts = predict_lstm(Nut_memory,values,features,nutrient,test_time,hyperparameters,ylim)
            # 5-fold cross validation was used to estimate the model performance
            print('Test RMSE: %.2f' % rmse)
            print('Test MAE: %.2f' % mae)
            print('Test R2: %.2f' %r2)
            Lake_Nut_metrics[nutrient]['MAE']=mae
            Lake_Nut_metrics[nutrient]['RMSE']=rmse
            Lake_Nut_metrics[nutrient]['R2']=r2
            print('Model takes '+str(round((time.time()-start_time)/60))+' min to run')
            figure.savefig(lakename+'_LSTM_'+nutrient+'.png',dpi = 500)    
            print('\n')
            Nut_memory = predict_ts(Nut_memory,nutrient,Nut_model,hyperparameters,values)
        print('Model takes '+str(round((time.time()-start_time)/60))+' min to run')
        Nut_memory.to_csv(lakename+'_LSTM predicted nutrient_'+str(i)+'.csv',sep='\t',index = False)
        Lake_Nut_metrics.to_csv(lakename+'_LSTM_metrics_'+str(i)+'.csv',index=False)
    else:
        break