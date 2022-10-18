#!/usr/bin/env python
# coding: utf-8

# # Import the data

# In[32]:


import pandas as pd # module that contain df methods 
import numpy as np # transforms a fd into an array
import math # the ability to do math equations in python
import matplotlib.pyplot as plt # module for simple data visualization for python 


# In[33]:


df = pd.read_csv('/Users/willjr/Desktop/Datasets/swedish_insurance.csv')


# In[34]:


df.describe()


# In[35]:


df.info()
# shows the data types of my df, and also informs me it contains have zero null values.


# In[36]:


print(df.columns)
df1 = df.head()
# we have 2 columns [X,Y] that we will rename into X: Claims Created, Y: Amount Paid.


# In[38]:


df_x = np.array(df['X'])
df_y = np.array(df['Y'])


# In[39]:


df_x


# In[40]:


df_y


# 
# # Visualize with histograms/Bar Chart

# In[24]:


# i want to show what my data looks like via histogram 


# In[41]:


plt.hist(df_x, bins=13, range=None)
# not very detailed, but we can build on this.
# below are the graphs for each column values for X, Y


# In[42]:


plt.hist(df_y, bins=20)


# In[43]:


# the lines of code below calculate mean and variance for both X and Y. 
mean_x = np.mean(df['X'])
mean_y = np.mean(df['Y'])

var_x = np.var(df['X'])
var_y = np.var(df['Y'])


print('x stats: mean= %.3f   variance= %.3f' % (mean_x, var_x))
print('y stats: mean= %.3f   variance= %.3f' % (mean_y, var_y))


# In[44]:


# Calculate covariance between x and y
def covariance(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar/len(x)



covar_xy = covariance(df['X'], df['Y'])
print(f'Cov(X,Y): {covar_xy}')


# In[45]:


b1 = covar_xy / var_x
b0 = mean_y - b1 * mean_x

print(f'Coefficents:\n b0: {b0}  b1: {b1} ')


# In[48]:


x = df['X'].values.copy()
x


# In[50]:


# Taking the values from the dataframe and sorting only X for the ease of plotting line later on
x = df['X'].values.copy()
# x.sort()
print(f'x: {x}')

# Predicting the new data based on calculated coeffiecents. 
y_hat = b0 + b1 * x
print(f'\n\ny_hat: {y_hat}')

y = df['Y'].values
print(f'\n\ny: {y}')


# In[51]:


len(y_hat)


# In[53]:


import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['X'], y=df['Y'], name='train', mode='markers', marker_color='rgba(152, 0, 0, .8)'))
fig.add_trace(go.Scatter(x=df['X'], y=y_hat, name='prediction', mode='lines+markers', marker_color='rgba(0, 152, 0, .8)'))

fig.update_layout(title = f'Swedish Automobiles Data\n (visual comparison for correctness)',title_x=0.5, xaxis_title= "Number of Claims", yaxis_title="Payment in Claims")
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.show()


# In[58]:


from sklearn.metrics import mean_squared_error
  
# Given values
Y_true = y  # Y_true = Y (original values)
  
# calculated values
Y_pred = y_hat  # Y_pred = Y'
  
# Calculation of Mean Squared Error (MSE)
mean_squared_error(Y_true,Y_pred)


# In[ ]:




