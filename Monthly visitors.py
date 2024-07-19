#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt


newquay_file = "Newquay Zoo Footfall 2018 2024.xlsx"
paignton_file = "Paignton Zoo Footfall 2018 2024.xlsx"


newquay_footfall = pd.read_excel(newquay_file, sheet_name='Newquay Footfall')
paignton_footfall = pd.read_excel(paignton_file, sheet_name='Paignton Footfall')


print(newquay_footfall.head())
print(paignton_footfall.head())


if 'Date' not in newquay_footfall.columns or 'Date' not in paignton_footfall.columns:
    raise KeyError("'Date' column not found in one of the DataFrames")


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


# In[12]:


print(newquay_footfall.head())
print(paignton_footfall.head())


if 'Date' not in newquay_footfall.columns or 'Date' not in paignton_footfall.columns:
    raise KeyError("'Date' column not found in one of the DataFrames")


# In[14]:


import pandas as pd

# File paths
newquay_file = "Newquay Zoo Footfall 2018 2024.xlsx"
paignton_file = "Paignton Zoo Footfall 2018 2024.xlsx"


newquay_footfall = pd.read_excel(newquay_file, sheet_name='Newquay Footfall')
paignton_footfall = pd.read_excel(paignton_file, sheet_name='Paignton Footfall')


print("Newquay Footfall Columns:", newquay_footfall.columns)
print("Paignton Footfall Columns:", paignton_footfall.columns)


print(newquay_footfall.head())
print(paignton_footfall.head())


newquay_footfall.columns = newquay_footfall.columns.str.strip()
paignton_footfall.columns = paignton_footfall.columns.str.strip()

if 'Date' not in newquay_footfall.columns or 'Date' not in paignton_footfall.columns:
    raise KeyError("'Date' column not found in one of the DataFrames")

newquay_footfall['Date'] = pd.to_datetime(newquay_footfall['Date'])
paignton_footfall['Date'] = pd.to_datetime(paignton_footfall['Date'])


newquay_footfall.set_index('Date', inplace=True)
paignton_footfall.set_index('Date', inplace=True)


newquay_monthly = newquay_footfall['Total'].resample('M').sum()
paignton_monthly = paignton_footfall['Total'].resample('M').sum()


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))

plt.plot(newquay_monthly, label='Newquay Zoo')
plt.plot(paignton_monthly, label='Paignton Zoo')


plt.xlim([pd.Timestamp('2019-01-01'), pd.Timestamp('2024-01-01')])

plt.title('Monthly Total Visitors: Newquay Zoo vs Paignton Zoo')
plt.xlabel('Date')
plt.ylabel('Total Visitors')
plt.legend()
plt.grid(True)
plt.show()


# In[8]:





# In[ ]:




