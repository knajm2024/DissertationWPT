#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


newquay_footfall = pd.read_excel('Newquay Zoo Footfall 2018 2024.xlsx')
paignton_footfall = pd.read_excel('Paignton Zoo Footfall 2018 2024.xlsx')

newquay_footfall.head(), paignton_footfall.head()


# In[2]:


newquay_sheets = pd.ExcelFile('Newquay Zoo Footfall 2018 2024.xlsx')
paignton_sheets = pd.ExcelFile('Paignton Zoo Footfall 2018 2024.xlsx')


newquay_sheets.sheet_names, paignton_sheets.sheet_names


# In[3]:


newquay_footfall = pd.read_excel('Newquay Zoo Footfall 2018 2024.xlsx', sheet_name='Newquay Footfall')
paignton_footfall = pd.read_excel('Paignton Zoo Footfall 2018 2024.xlsx', sheet_name='Paignton Footfall')


newquay_footfall.head(), paignton_footfall.head()


# In[8]:


import matplotlib.pyplot as plt
import pandas as pd


newquay_footfall['Date'] = pd.to_datetime(newquay_footfall['Date'])
paignton_footfall['Date'] = pd.to_datetime(paignton_footfall['Date'])


newquay_footfall.set_index('Date', inplace=True)
paignton_footfall.set_index('Date', inplace=True)


newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()


plt.figure(figsize=(14, 7))

plt.plot(newquay_monthly, label='Newquay Zoo')
plt.plot(paignton_monthly, label='Paignton Zoo')

plt.xlim(['2019', '2024'])

plt.title('Monthly Total Visitors: Newquay Zoo vs Paignton Zoo')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import pandas as pd


if 'Date' not in newquay_footfall.columns or 'Date' not in paignton_footfall.columns:
    raise KeyError("'Date' column not found in one of the DataFrames")


newquay_footfall['Date'] = pd.to_datetime(newquay_footfall['Date'])
paignton_footfall['Date'] = pd.to_datetime(paignton_footfall['Date'])


newquay_footfall.set_index('Date', inplace=True)
paignton_footfall.set_index('Date', inplace=True)


newquay_monthly = newquay_footfall.resample('M')['Total'].sum()
paignton_monthly = paignton_footfall.resample('M')['Total'].sum()

plt.figure(figsize=(14, 7))

plt.plot(newquay_monthly, label='Newquay Zoo')
plt.plot(paignton_monthly, label='Paignton Zoo')


plt.xlim(['2019', '2024'])

plt.title('Monthly Total Visitors: Newquay Zoo vs Paignton Zoo')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

newquay_footfall['Date'] = pd.to_datetime(newquay_footfall['Date'])
paignton_footfall['Date'] = pd.to_datetime(paignton_footfall['Date'])


newquay_footfall.set_index('Date', inplace=True)
paignton_footfall.set_index('Date', inplace=True)


newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()


plt.figure(figsize=(14, 7))

plt.plot(newquay_monthly, label='Newquay Zoo')
plt.plot(paignton_monthly, label='Paignton Zoo')


plt.xlim(['2019', '2024'])

plt.title('Monthly Total Visitors: Newquay Zoo vs Paignton Zoo')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns



newquay_footfall.replace('-', 0, inplace=True)
paignton_footfall.replace('-', 0, inplace=True)

newquay_clustering_data = newquay_footfall[['Admissions\nAdult', 'Admissions\nChild', 'Education\nAdult', 'Education\nChild', 'Family\nAdult', 'Family\nChild', 'Unpaids\nAdult', 'Unpaids\nChild', 'Annual Members\nAdult', 'Annual Members\nChild']].apply(pd.to_numeric, errors='coerce').fillna(0)
paignton_clustering_data = paignton_footfall[['Admissions\nAdult', 'Admissions\nChild', 'Education\nAdult', 'Education\nChild', 'Family\nAdult', 'Family\nChild', 'Unpaids\nAdult', 'Unpaids\nChild', 'Annual Members\nAdult', 'Annual Members\nChild']].apply(pd.to_numeric, errors='coerce').fillna(0)


scaler = StandardScaler()
newquay_scaled = scaler.fit_transform(newquay_clustering_data)
paignton_scaled = scaler.fit_transform(paignton_clustering_data)

kmeans_newquay = KMeans(n_clusters=3, random_state=42)
kmeans_paignton = KMeans(n_clusters=3, random_state=42)

newquay_clusters = kmeans_newquay.fit_predict(newquay_scaled)
paignton_clusters = kmeans_paignton.fit_predict(paignton_scaled)


newquay_footfall['Cluster'] = newquay_clusters
paignton_footfall['Cluster'] = paignton_clusters


combined_footfall = pd.concat([newquay_footfall, paignton_footfall], keys=['Newquay', 'Paignton']).reset_index(level=0).rename(columns={'level_0': 'Zoo'})


plt.figure(figsize=(14, 7))
sns.scatterplot(data=combined_footfall, x='Admissions\nAdult', y='Admissions\nChild', hue='Cluster', style='Zoo', palette='viridis')
plt.title('Visitor Clusters for Newquay and Paignton Zoos')
plt.xlabel('Admissions (Adult)')
plt.ylabel('Admissions (Child)')
plt.legend(loc='best', title='Cluster')
plt.grid(True)
plt.show()


# In[14]:


plt.figure(figsize=(14, 14))

plt.subplot(2, 1, 1)
sns.scatterplot(data=newquay_footfall, x='Admissions\nAdult', y='Admissions\nChild', hue='Cluster', palette='viridis')
plt.title('Visitor Clusters for Newquay Zoo')
plt.xlabel('Admissions (Adult)')
plt.ylabel('Admissions (Child)')
plt.legend(loc='best', title='Cluster')
plt.grid(True)


plt.subplot(2, 1, 2)
sns.scatterplot(data=paignton_footfall, x='Admissions\nAdult', y='Admissions\nChild', hue='Cluster', palette='viridis')
plt.title('Visitor Clusters for Paignton Zoo')
plt.xlabel('Admissions (Adult)')
plt.ylabel('Admissions (Child)')
plt.legend(loc='best', title='Cluster')
plt.grid(True)

plt.tight_layout()
plt.show()


# In[15]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Prepare the data for ARIMA
newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()

model_newquay = SARIMAX(newquay_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_paignton = SARIMAX(paignton_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

result_newquay = model_newquay.fit(disp=False)
result_paignton = model_paignton.fit(disp=False)


forecast_newquay = result_newquay.get_forecast(steps=12)
forecast_paignton = result_paignton.get_forecast(steps=12)


forecast_newquay_df = forecast_newquay.conf_int()
forecast_newquay_df['forecast'] = forecast_newquay.predicted_mean

forecast_paignton_df = forecast_paignton.conf_int()
forecast_paignton_df['forecast'] = forecast_paignton.predicted_mean


forecast_newquay_df = forecast_newquay_df.apply(pd.to_numeric, errors='coerce')
forecast_paignton_df = forecast_paignton_df.apply(pd.to_numeric, errors='coerce')

# Plot the forecasts
plt.figure(figsize=(14, 7))


plt.subplot(2, 1, 1)
plt.plot(newquay_monthly.index, newquay_monthly, label='Actual')
plt.plot(forecast_newquay_df.index, forecast_newquay_df['forecast'], label='Forecast')
plt.fill_between(forecast_newquay_df.index, forecast_newquay_df.iloc[:, 0], forecast_newquay_df.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Newquay Zoo Monthly Visitors Forecast')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()

# Plot Paignton Zoo forecast
plt.subplot(2, 1, 2)
plt.plot(paignton_monthly.index, paignton_monthly, label='Actual')
plt.plot(forecast_paignton_df.index, forecast_paignton_df['forecast'], label='Forecast')
plt.fill_between(forecast_paignton_df.index, forecast_paignton_df.iloc[:, 0], forecast_paignton_df.iloc[:, 1], color='lightblue', alpha=0.3)
plt.title('Paignton Zoo Monthly Visitors Forecast')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()

plt.tight_layout()
plt.show()


# In[16]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prepare the data for ARIMA
newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()


model_newquay = SARIMAX(newquay_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_paignton = SARIMAX(paignton_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

result_newquay = model_newquay.fit(disp=False)
result_paignton = model_paignton.fit(disp=False)

forecast_newquay = result_newquay.get_forecast(steps=24)
forecast_paignton = result_paignton.get_forecast(steps=24)

forecast_newquay_df = forecast_newquay.conf_int()
forecast_newquay_df['forecast'] = forecast_newquay.predicted_mean

forecast_paignton_df = forecast_paignton.conf_int()
forecast_paignton_df['forecast'] = forecast_paignton.predicted_mean


forecast_newquay_df = forecast_newquay_df.apply(pd.to_numeric, errors='coerce')
forecast_paignton_df = forecast_paignton_df.apply(pd.to_numeric, errors='coerce')


plt.figure(figsize=(14, 7))

# Plot Newquay Zoo forecast
plt.subplot(2, 1, 1)
plt.plot(newquay_monthly.index, newquay_monthly, label='Actual')
plt.plot(forecast_newquay_df.index, forecast_newquay_df['forecast'], label='Forecast')
plt.fill_between(forecast_newquay_df.index, forecast_newquay_df.iloc[:, 0], forecast_newquay_df.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Newquay Zoo Monthly Visitors Forecast for 2024 and 2025')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()

# Plot Paignton Zoo forecast
plt.subplot(2, 1, 2)
plt.plot(paignton_monthly.index, paignton_monthly, label='Actual')
plt.plot(forecast_paignton_df.index, forecast_paignton_df['forecast'], label='Forecast')
plt.fill_between(forecast_paignton_df.index, forecast_paignton_df.iloc[:, 0], forecast_paignton_df.iloc[:, 1], color='lightblue', alpha=0.3)
plt.title('Paignton Zoo Monthly Visitors Forecast for 2024 and 2025')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()

plt.tight_layout()
plt.show()


# In[18]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()


model_newquay = SARIMAX(newquay_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_paignton = SARIMAX(paignton_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

result_newquay = model_newquay.fit(disp=False)
result_paignton = model_paignton.fit(disp=False)


forecast_newquay = result_newquay.get_forecast(steps=12)
forecast_paignton = result_paignton.get_forecast(steps=12)


forecast_newquay_df = forecast_newquay.conf_int()
forecast_newquay_df['forecast'] = forecast_newquay.predicted_mean

forecast_paignton_df = forecast_paignton.conf_int()
forecast_paignton_df['forecast'] = forecast_paignton.predicted_mean

# Convert to numeric values
forecast_newquay_df = forecast_newquay_df.apply(pd.to_numeric, errors='coerce')
forecast_paignton_df = forecast_paignton_df.apply(pd.to_numeric, errors='coerce')


plt.figure(figsize=(14, 7))


plt.subplot(2, 1, 1)
plt.plot(newquay_monthly.index, newquay_monthly, label='Actual')
plt.plot(forecast_newquay_df.index, forecast_newquay_df['forecast'], label='Forecast')
plt.fill_between(forecast_newquay_df.index, forecast_newquay_df['lower Total'], forecast_newquay_df['upper Total'], color='pink', alpha=0.3)
plt.title('Newquay Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(newquay_monthly.index.min(), pd.Timestamp('2024-12-31'))
plt.legend()

# Plot Paignton Zoo forecast
plt.subplot(2, 1, 2)
plt.plot(paignton_monthly.index, paignton_monthly, label='Actual')
plt.plot(forecast_paignton_df.index, forecast_paignton_df['forecast'], label='Forecast')
plt.fill_between(forecast_paignton_df.index, forecast_paignton_df['lower Total'], forecast_paignton_df['upper Total'], color='lightblue', alpha=0.3)
plt.title('Paignton Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(paignton_monthly.index.min(), pd.Timestamp('2024-12-31'))
plt.legend()

plt.tight_layout()
plt.show()


# In[20]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()

hw_model_newquay = ExponentialSmoothing(newquay_monthly, seasonal='add', seasonal_periods=12).fit()
hw_model_paignton = ExponentialSmoothing(paignton_monthly, seasonal='add', seasonal_periods=12).fit()


hw_forecast_newquay = hw_model_newquay.forecast(steps=12)
hw_forecast_paignton = hw_model_paignton.forecast(steps=12)


hw_forecast_newquay_df = pd.DataFrame(hw_forecast_newquay, columns=['forecast'])
hw_forecast_newquay_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')

hw_forecast_paignton_df = pd.DataFrame(hw_forecast_paignton, columns=['forecast'])
hw_forecast_paignton_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')

plt.figure(figsize=(14, 7))


plt.subplot(2, 1, 1)
plt.plot(newquay_monthly.index, newquay_monthly, label='Actual')
plt.plot(hw_forecast_newquay_df.index, hw_forecast_newquay_df['forecast'], label='Forecast', color='red')
plt.title('Newquay Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(newquay_monthly.index.min(), pd.Timestamp('2024-12-31'))
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(paignton_monthly.index, paignton_monthly, label='Actual')
plt.plot(hw_forecast_paignton_df.index, hw_forecast_paignton_df['forecast'], label='Forecast', color='red')
plt.title('Paignton Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(paignton_monthly.index.min(), pd.Timestamp('2024-12-31'))
plt.legend()

plt.tight_layout()
plt.show()


# In[21]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prepare the data for ARIMA
newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()

# Fit ARIMA model
model_newquay = SARIMAX(newquay_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
model_paignton = SARIMAX(paignton_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)

result_newquay = model_newquay.fit(disp=False)
result_paignton = model_paignton.fit(disp=False)

forecast_newquay = result_newquay.get_forecast(steps=12)
forecast_paignton = result_paignton.get_forecast(steps=12)

forecast_newquay_df = forecast_newquay.conf_int()
forecast_newquay_df['forecast'] = forecast_newquay.predicted_mean

forecast_paignton_df = forecast_paignton.conf_int()
forecast_paignton_df['forecast'] = forecast_paignton.predicted_mean


forecast_newquay_df = forecast_newquay_df.apply(pd.to_numeric, errors='coerce')
forecast_paignton_df = forecast_paignton_df.apply(pd.to_numeric, errors='coerce')


forecast_newquay_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')
forecast_paignton_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')


plt.figure(figsize=(14, 7))


plt.subplot(2, 1, 1)
plt.plot(newquay_monthly.index, newquay_monthly, label='Actual')
plt.plot(forecast_newquay_df.index, forecast_newquay_df['forecast'], label='Forecast', color='red')
plt.fill_between(forecast_newquay_df.index, forecast_newquay_df['lower Total'], forecast_newquay_df['upper Total'], color='pink', alpha=0.3)
plt.title('Newquay Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(newquay_monthly.index.min(), pd.Timestamp('2024-12-31'))
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(paignton_monthly.index, paignton_monthly, label='Actual')
plt.plot(forecast_paignton_df.index, forecast_paignton_df['forecast'], label='Forecast', color='red')
plt.fill_between(forecast_paignton_df.index, forecast_paignton_df['lower Total'], forecast_paignton_df['upper Total'], color='lightblue', alpha=0.3)
plt.title('Paignton Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(paignton_monthly.index.min(), pd.Timestamp('2024-12-31'))
plt.legend()

plt.tight_layout()
plt.show()


# In[22]:


from statsmodels.tsa.seasonal import seasonal_decompose

decompose_newquay = seasonal_decompose(newquay_monthly, model='additive')
decompose_paignton = seasonal_decompose(paignton_monthly, model='additive')

#the decomposition
plt.figure(figsize=(14, 10))

#decomposition for Newquay Zoo
plt.subplot(3, 2, 1)
plt.plot(decompose_newquay.trend)
plt.title('Newquay Zoo Trend')

plt.subplot(3, 2, 3)
plt.plot(decompose_newquay.seasonal)
plt.title('Newquay Zoo Seasonality')

plt.subplot(3, 2, 5)
plt.plot(decompose_newquay.resid)
plt.title('Newquay Zoo Residuals')

# decomposition for Paignton Zoo
plt.subplot(3, 2, 2)
plt.plot(decompose_paignton.trend)
plt.title('Paignton Zoo Trend')

plt.subplot(3, 2, 4)
plt.plot(decompose_paignton.seasonal)
plt.title('Paignton Zoo Seasonality')

plt.subplot(3, 2, 6)
plt.plot(decompose_paignton.resid)
plt.title('Paignton Zoo Residuals')

plt.tight_layout()
plt.show()


# In[24]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt


newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()


model_newquay = SARIMAX(newquay_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
model_paignton = SARIMAX(paignton_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)

result_newquay = model_newquay.fit(disp=False)
result_paignton = model_paignton.fit(disp=False)

forecast_newquay = result_newquay.get_forecast(steps=12)
forecast_paignton = result_paignton.get_forecast(steps=12)


forecast_newquay_df = forecast_newquay.conf_int()
forecast_newquay_df['forecast'] = forecast_newquay.predicted_mean

forecast_paignton_df = forecast_paignton.conf_int()
forecast_paignton_df['forecast'] = forecast_paignton.predicted_mean


forecast_newquay_df = forecast_newquay_df.apply(pd.to_numeric, errors='coerce')
forecast_paignton_df = forecast_paignton_df.apply(pd.to_numeric, errors='coerce')


forecast_newquay_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')
forecast_paignton_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')


plt.figure(figsize=(14, 7))


plt.subplot(2, 1, 1)
plt.plot(newquay_monthly.index, newquay_monthly, label='Actual')
plt.plot(forecast_newquay_df.index, forecast_newquay_df['forecast'], label='Forecast', color='red')
plt.fill_between(forecast_newquay_df.index, forecast_newquay_df['lower Total'], forecast_newquay_df['upper Total'], color='pink', alpha=0.3)
plt.title('Newquay Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2024-12-31'))
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(paignton_monthly.index, paignton_monthly, label='Actual')
plt.plot(forecast_paignton_df.index, forecast_paignton_df['forecast'], label='Forecast', color='red')
plt.fill_between(forecast_paignton_df.index, forecast_paignton_df['lower Total'], forecast_paignton_df['upper Total'], color='lightblue', alpha=0.3)
plt.title('Paignton Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2024-12-31'))
plt.legend()

plt.tight_layout()
plt.show()


# In[25]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prepare the data for Holt-Winters
newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()

# Fit Holt-Winters model
hw_model_newquay = ExponentialSmoothing(newquay_monthly, trend='add', seasonal='add', seasonal_periods=12).fit()
hw_model_paignton = ExponentialSmoothing(paignton_monthly, trend='add', seasonal='add', seasonal_periods=12).fit()


hw_forecast_newquay = hw_model_newquay.forecast(steps=12)
hw_forecast_paignton = hw_model_paignton.forecast(steps=12)


hw_forecast_newquay_df = pd.DataFrame(hw_forecast_newquay, columns=['forecast'])
hw_forecast_newquay_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')

hw_forecast_paignton_df = pd.DataFrame(hw_forecast_paignton, columns=['forecast'])
hw_forecast_paignton_df.index = pd.date_range(start='2024-01-01', periods=12, freq='M')


plt.figure(figsize=(14, 7))


plt.subplot(2, 1, 1)
plt.plot(newquay_monthly.index, newquay_monthly, label='Actual')
plt.plot(hw_forecast_newquay_df.index, hw_forecast_newquay_df['forecast'], label='Forecast', color='red')
plt.title('Newquay Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2024-12-31'))
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(paignton_monthly.index, paignton_monthly, label='Actual')
plt.plot(hw_forecast_paignton_df.index, hw_forecast_paignton_df['forecast'], label='Forecast', color='red')
plt.title('Paignton Zoo Monthly Visitors Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2024-12-31'))
plt.legend()

plt.tight_layout()
plt.show()


# In[28]:


import pandas as pd
import matplotlib.pyplot as plt


newquay_monthly_totals = newquay_footfall['Total'].resample('M').sum()
paignton_monthly_totals = paignton_footfall['Total'].resample('M').sum()


newquay_peak_months = newquay_monthly_totals.nlargest(5)
paignton_peak_months = paignton_monthly_totals.nlargest(5)


plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.bar(newquay_peak_months.index.strftime('%Y-%m'), newquay_peak_months.values, color='blue')
plt.title('Peak Visitor Months for Newquay Zoo')
plt.xlabel('Month')
plt.ylabel('Total Visitors')


plt.subplot(2, 1, 2)
plt.bar(paignton_peak_months.index.strftime('%Y-%m'), paignton_peak_months.values, color='green')
plt.title('Peak Visitor Months for Paignton Zoo')
plt.xlabel('Month')
plt.ylabel('Total Visitors')

plt.tight_layout()
plt.show()


# In[ ]:




