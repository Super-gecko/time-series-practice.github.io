#!/usr/bin/env python
# coding: utf-8

#1. Read Data

import pandas as pd

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


df = pd.read_csv('temp_datalab_records_social_facebook.csv')
df.head()

df.shape

df.time.max()

df.time.min()

print('This dataset has ~3.6 million data points from 01-01-2015 to 07-17-2018')

# ### 3 Data preprocessing

# #### a. Check and fill missing values in username

df_new = df[df.username.notnull()]
df_new.facebook_id.nunique()
df_new.username.nunique()

## make a dictionary with two columns from the dataframe

dict_name_id = pd.Series(df_new.username.values,index=df_new.facebook_id).to_dict()

# create a new username column by mapping the values from the disctionary that is just created, where the keys of the dictionary is kept as same as the facebook_id column of that dataframe
df['username_re'] = df['facebook_id'].map(dict_name_id)
df.username_re.nunique()

df_remove_na = df.loc[df.username_re.notnull()]

df_remove_na.shape

print('Dataframe df_remove_na and column username_re will be used in the following analysis')


# #### b.check missing values in df_remove_na
#  entity_id cusip isin are columns of nulls and these can be removed.

print(df_remove_na.isnull().sum())

df_remove_na = df_remove_na.drop(['entity_id', 'cusip','isin'], axis=1)
print('remove null columns: entity_id, cusip and isin')

# #### c. check the correlation

cor = df_remove_na.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(9, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor, mask=mask, cmap=cmap, vmax=.8, vmin = -.6, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# ###  4. Exploratory Data Analysis
# 

# a. Companies with highest volatility of “talking about count” 

df_a = df_remove_na.groupby('username_re').talking_about_count.mean().reset_index().sort_values(['talking_about_count'], ascending=0).reset_index()
df_a.username_re[:10]

fig, ax = plt.subplots(figsize = ( 10, 6 ))
sns.barplot(x = 'username_re', y = 'talking_about_count', saturation = 0.5, data = df_a.head(10))
plt.title('Top 10 companies with highest volatility of “talking about count”', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Companies', fontsize = 24)
plt.ylabel('talking_about_count', fontsize= 24)
plt.ticklabel_format(style='scientific', axis='y')

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25
#ax.ticklabel_format(axis='y', style='scientific')


plt.show()


# b. Winners in attracting customers to physical locations.

df_b = df_remove_na.groupby('username_re').checkins.mean().reset_index().sort_values(['checkins'], ascending=0).reset_index()
df_b.username_re[:10]

fig, ax = plt.subplots(figsize = ( 10, 6 ))

sns.barplot(x = 'username_re', y = 'checkins', saturation = 0.5, data = df_b.head(10))
plt.title('Top 10 companies in attracting customers to physical locations', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Companies', fontsize = 24)
plt.ylabel('checkins', fontsize= 24)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

plt.show()

fig, ax = plt.subplots(figsize = ( 20, 8 ))
sns.barplot(x = 'username_re', y = 'checkins', saturation = 0.5, data = df_b.head(50))
plt.title('Top 50 companies in attracting customers to physical locations', fontsize = 24)
plt.xticks(rotation=45, fontsize = 16, ha='right')
plt.yticks(fontsize = 16)
plt.xlabel('Companies', fontsize = 24)
plt.ylabel('checkins', fontsize= 24)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

plt.show()


# c. Facebook followers and which companies are the most successful at growing social media traction

df_c = df_remove_na.groupby('username_re').likes.mean().reset_index().sort_values(['likes'], ascending=0).reset_index()
df_c.username_re[:10]

fig, ax = plt.subplots(figsize = ( 10, 6 ))

sns.barplot(x = 'username_re', y = 'likes', saturation = 0.5, data = df_c.head(10))
plt.title('Top 10 companies that are the most successful at growing social media traction', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Companies', fontsize = 24)
plt.ylabel('Likes', fontsize= 24)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25

plt.show()

# d. were_here_count, Could it also be used as a metric for foot traffic?

df_d = df_remove_na.groupby('username_re').were_here_count.mean().reset_index().sort_values(['were_here_count'], ascending=0).reset_index()
df_d.username_re[:10]

fig, ax = plt.subplots(figsize = ( 10, 6 ))

sns.barplot(x = 'username_re', y = 'were_here_count', saturation = 0.5, data = df_d.head(10))
plt.title('Top 10 companies in attracting customers to physical locations metric2', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Companies', fontsize = 22)
plt.ylabel('were_here_count', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25

plt.show()

df_remove_na.head()

# ### 5.  Time Series Forescasting

# convert string 'time' into datetime format, if there is any 
df_remove_na['time']= pd.to_datetime(df['time']) 
# extract the date info from the time column, and store it into additional column called 'date'; and ignore the hours
df_remove_na['date'] = [d.date() for d in df_remove_na['time']]

#Take out data points only for Disneyland to make time series
disneyland_ts = df_remove_na[df_remove_na.username_re == 'Disneyland'][['date','checkins']]

disneyland_ts.index = pd.to_datetime(disneyland_ts['date'])
#df = df.set_index('datetime')
disneyland_ts.drop(['date'], axis=1, inplace=True)
disneyland_ts.head()


diff_ts=disneyland_ts.diff()
diff_ts.head()


# Plot the time series just created
fig, ax = plt.subplots(figsize = ( 18, 6 ))

plt.plot(diff_ts)

plt.title('Daily net increase in foot traffic at Disneyland', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Foot traffic', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

# #### Check Stationary 
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(10).mean()
    rolstd = timeseries.rolling(10).std()

    #Plot rolling statistics:
    
    fig, ax = plt.subplots(figsize = ( 18, 6 ))

    
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best', fontsize = 20 )
    plt.title('Rolling Mean & Standard Deviation', fontsize = 24)
    
    plt.xticks(rotation=45, fontsize = 20, ha='right')
    plt.yticks(fontsize = 20)
    plt.xlabel('Time', fontsize = 22)
    plt.ylabel('Foot traffic', fontsize= 22)
    
    ax.title.set_position([.5, 1.1])
    ax.yaxis.labelpad = 25
    ax.xaxis.labelpad = 25
    
    plt.show(block=False)

test_stationarity(diff_ts)

# Estimating & Eliminating Trend
ts_log = np.log(diff_ts)

fig, ax = plt.subplots(figsize = ( 18, 6 ))
plt.plot(ts_log)

plt.title('Daily net increase in foot traffic at Disneyland(Log transform)', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Foot traffic', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

ts_log.dropna(inplace=True)
test_stationarity(ts_log)


# Eliminating Trend and Seasonality

decomposition = seasonal_decompose(ts_log, freq = 52)

Trend = decomposition.trend
Seasonal = decomposition.seasonal
Residual = decomposition.resid

fig, ax = plt.subplots(figsize = ( 18, 18 ))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
    
plt.subplot(4,1,1)
plt.plot(ts_log, label= 'Original')
plt.legend(loc='best', fontsize = 20)

plt.title('Original', fontsize = 24)
#plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
#plt.xlabel('Time', fontsize = 22)
plt.ylabel('Foot traffic', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
#ax.xaxis.labelpad = 25

plt.subplot(4,1,2)
plt.plot(Trend, label= 'Trend')
plt.legend(loc='best', fontsize = 20)

plt.title('Trend', fontsize = 24)
#plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
#plt.xlabel('Time', fontsize = 22)
plt.ylabel('Foot traffic', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
#ax.xaxis.labelpad = 25

plt.subplot(4,1,3)
plt.plot(Seasonal, label= 'Seasonal')
plt.legend(loc='best', fontsize = 20)

plt.title('Seasonal', fontsize = 24)
#plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
#plt.xlabel('Time', fontsize = 22)

plt.ylabel('Foot traffic', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
#ax.xaxis.labelpad = 25

plt.subplot(4,1,4)
plt.plot(Residual, label= 'Residual')
plt.legend(loc='best', fontsize = 20)

plt.title('Residual', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Foot traffic', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

#plt.tight_layout()
fig.suptitle('Decomposing', fontsize= 25)


plt.show()


# Here we can see that the trend, seasonality are separated out from data and we can model the residuals. Lets check stationarity of residuals:

ts_log_decompose = Residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

# #### Differencing

ts_log_diff = ts_log - ts_log.shift()

fig, ax = plt.subplots(figsize = ( 18, 6 ))

plt.plot(ts_log_diff)

plt.title('Differencing', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Foot traffic', fontsize= 22)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# We can see that the mean and std variations have small variations with time. 

# ### Forecasting a Time Series

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf


lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 
fig, ax = plt.subplots(figsize = ( 18, 10 ))

plt.subplot(121)
plt.plot(lag_acf)

plt.title('Autocorrelation Function', fontsize = 24)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Correlation Time', fontsize = 22)
plt.ylabel('ACF', fontsize= 22)

ax.title.set_position([.5, 1.05])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)

plt.title('Partial Autocorrelation Function', fontsize = 24)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Correlation Time', fontsize = 22)
plt.ylabel('PACF', fontsize= 22)

ax.title.set_position([.5, 1.05])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.show()

# In this plot, the two dotted lines on either sides of 0 are the confidence interevals. These can be used to determine the ‘p’ and ‘q’ values as:
# 
#     p – The lag value where the PACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case p=1.
#     q – The lag value where the ACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case q=1.

# ### AR Model

model = ARIMA(ts_log, order=(1, 1, 0))  
results_AR = model.fit(disp=-1) 

df_RSS = ts_log_diff[['checkins']]
df_RSS['fittedvalues']=results_AR.fittedvalues
df_RSS['delta']= df_RSS.checkins-df_RSS.fittedvalues
RSS_AR = df_RSS.delta.pow(2).sum()
print('RSS: ', RSS_AR)

ts_log_diff.head()

fig, ax = plt.subplots(figsize = ( 18, 6 ))

plt.plot(ts_log_diff, label= 'Original')
plt.plot(results_AR.fittedvalues, label= 'Predicted')

plt.title('Auto-Regressive Model Results', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Time', fontsize = 22)
#plt.ylabel('Foot traffic', fontsize= 22)
plt.legend(loc='best', fontsize = 20)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

### MA Model

model = ARIMA(ts_log, order=(0, 1, 1))  
results_MA = model.fit(disp=-1)  


MA_RSS = ts_log_diff[['checkins']]
MA_RSS['fittedvalues']=results_MA.fittedvalues
MA_RSS['delta']= MA_RSS.checkins-MA_RSS.fittedvalues
RSS_MA = MA_RSS.delta.pow(2).sum()
print('RSS: ', RSS_MA)


fig, ax = plt.subplots(figsize = ( 18, 6 ))

plt.plot(ts_log_diff, label= 'Original')
plt.plot(results_MA.fittedvalues, label= 'Predicted')

plt.title('Moving Averages Model Results', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Time', fontsize = 22)
#plt.ylabel('Foot traffic', fontsize= 22)
plt.legend(loc='best', fontsize = 20)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

### ARIMA Model(Combined Model)

model = ARIMA(ts_log, order=(1, 1, 1))  
results_ARIMA = model.fit(disp=-1)  


ARIMA_RSS = ts_log_diff[['checkins']]
ARIMA_RSS['fittedvalues']=results_ARIMA.fittedvalues
ARIMA_RSS['delta']= ARIMA_RSS.checkins-ARIMA_RSS.fittedvalues
RSS_ARIMA = MA_RSS.delta.pow(2).sum()
print('RSS: ', RSS_ARIMA)

fig, ax = plt.subplots(figsize = ( 18, 6 ))

plt.plot(ts_log_diff, label= 'Original')
plt.plot(results_ARIMA.fittedvalues, label= 'Predicted')

plt.title('ARIMA Model Results', fontsize = 24)
plt.xticks(rotation=45, fontsize = 20, ha='right')
plt.yticks(fontsize = 20)
plt.xlabel('Time', fontsize = 22)
#plt.ylabel('Foot traffic', fontsize= 22)
plt.legend(loc='best', fontsize = 20)

ax.title.set_position([.5, 1.1])
ax.yaxis.labelpad = 25
ax.xaxis.labelpad = 25

# ### Taking it back to original scale

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

# To convert the differencing to log scale is to add these differences consecutively to the base number. An easy way to do it is to first determine the cumulative sum at index and then add it to the base number. The cumulative sum can be found as:

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# In[ ]:




