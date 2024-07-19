#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Data
data = {
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Income': [10.5, 11.0, 10.7, 12.1, 11.8],
    'Expenditure': [9.8, 10.2, 10.0, 10.7, 12.0],
    'Net Income': [0.7, 0.8, 0.7, 1.4, -0.2],
    'Employment': [280, 290, 270, 267, 296],
    'Visitors': [450000, 460000, 440000, 520000, 482153]
}

df = pd.DataFrame(data)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
plt.plot(df['Year'], df['Income'], label='Income')
plt.plot(df['Year'], df['Expenditure'], label='Expenditure')
plt.xlabel('Year')
plt.ylabel('Amount (£ million)')
plt.title('Income and Expenditure Over Time')
plt.xticks(df['Year'])  
plt.legend()


plt.subplot(2, 2, 2)
plt.plot(df['Year'], df['Net Income'], label='Net Income', color='green')
plt.xlabel('Year')
plt.ylabel('Amount (£ million)')
plt.title('Net Income Over Time')
plt.xticks(df['Year'])  
plt.legend()


plt.subplot(2, 2, 3)
plt.plot(df['Year'], df['Employment'], label='Employment', color='orange')
plt.xlabel('Year')
plt.ylabel('Number of Employees')
plt.title('Employment Over Time')
plt.xticks(df['Year']) 
plt.legend()


plt.subplot(2, 2, 4)
plt.plot(df['Year'], df['Visitors'], label='Visitors', color='red')
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.title('Visitor Numbers Over Time')
plt.xticks(df['Year']) 
plt.legend()

plt.tight_layout()
plt.show()


# In[2]:


from sklearn.linear_model import LinearRegression
import numpy as np


X = df[['Year']]
y_income = df['Income']
y_visitors = df['Visitors']


model_income = LinearRegression().fit(X, y_income)
model_visitors = LinearRegression().fit(X, y_visitors)


future_years = np.array([[2023], [2024], [2025], [2026], [2027]])
income_predictions = model_income.predict(future_years)
visitor_predictions = model_visitors.predict(future_years)


predictions = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Income': income_predictions,
    'Predicted Visitors': visitor_predictions
})

predictions


# In[3]:


# Prepare data for additional metrics
y_expenditure = df['Expenditure']
y_net_income = df['Net Income']
y_employment = df['Employment']


model_expenditure = LinearRegression().fit(X, y_expenditure)
model_net_income = LinearRegression().fit(X, y_net_income)
model_employment = LinearRegression().fit(X, y_employment)


expenditure_predictions = model_expenditure.predict(future_years)
net_income_predictions = model_net_income.predict(future_years)
employment_predictions = model_employment.predict(future_years)

predictions_extended = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Income': income_predictions,
    'Predicted Expenditure': expenditure_predictions,
    'Predicted Net Income': net_income_predictions,
    'Predicted Employment': employment_predictions,
    'Predicted Visitors': visitor_predictions
})

predictions_extended


# In[6]:


average_spending_per_visitor = 50
df['Total Visitor Spending'] = df['Visitors'] * average_spending_per_visitor

# Predicted future visitor spending
predictions_extended['Total Visitor Spending'] = predictions_extended['Predicted Visitors'] * average_spending_per_visitor

df[['Year', 'Visitors', 'Total Visitor Spending']]
predictions_extended[['Year', 'Predicted Visitors', 'Total Visitor Spending']]


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Year": [2023, 2024, 2025, 2026, 2027],
    "Predicted Income": [12.33, 12.70, 13.07, 13.44, 13.81],
    "Predicted Expenditure": [12.01, 12.50, 12.99, 13.48, 13.97],
    "Predicted Net Income": [0.32, 0.20, 0.08, -0.04, -0.16],
    "Predicted Employment": [283.3, 284.2, 285.1, 286.0, 286.9],
    "Predicted Visitors": [507722.4, 520153.0, 532583.6, 545014.2, 557444.8]
}

df = pd.DataFrame(data)

# Plotting 
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

axs[0, 0].plot(df["Year"], df["Predicted Income"], marker='o', color='blue')
axs[0, 0].set_title('Predicted Income over Years')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Predicted Income')


axs[0, 1].plot(df["Year"], df["Predicted Expenditure"], marker='o', color='green')
axs[0, 1].set_title('Predicted Expenditure over Years')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Predicted Expenditure')


axs[1, 0].plot(df["Year"], df["Predicted Net Income"], marker='o', color='red')
axs[1, 0].set_title('Predicted Net Income over Years')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Predicted Net Income')


axs[1, 1].plot(df["Year"], df["Predicted Employment"], marker='o', color='purple')
axs[1, 1].set_title('Predicted Employment over Years')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Predicted Employment')


axs[2, 0].plot(df["Year"], df["Predicted Visitors"], marker='o', color='orange')
axs[2, 0].set_title('Predicted Visitors over Years')
axs[2, 0].set_xlabel('Year')
axs[2, 0].set_ylabel('Predicted Visitors')


fig.delaxes(axs[2, 1])

# Adjust layout
plt.tight_layout()
plt.show()


# In[7]:


# Historical Total Visitor Spending
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df['Year'], df['Total Visitor Spending'], label='Total Visitor Spending', color='green')
plt.xlabel('Year')
plt.ylabel('Total Visitor Spending (£)')
plt.title('Total Visitor Spending Over Time')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(predictions_extended['Year'], predictions_extended['Total Visitor Spending'], label='Predicted Total Visitor Spending', color='purple')
plt.xlabel('Year')
plt.ylabel('Predicted Total Visitor Spending (£)')
plt.title('Predicted Total Visitor Spending Over Time')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


All good

