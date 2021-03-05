#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[3]:


iris = datasets.load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)


# In[4]:


_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8, 8], s = 150, marker = 'D')


# In[6]:


from sklearn.neighbors import KNeighborsClassifier
df.head()


# In[7]:


knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X,y)
y_pred = knn.predict(X)


# In[16]:


# Predict and print the label for the new data point X_new
X_new = np.random.rand(150, 4) * 10
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))


# In[17]:


from sklearn import datasets
digits = datasets.load_digits()


# In[18]:


# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)


# In[19]:


# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[56]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# Create the prediction space
X = np.array(df.iloc[:,0]).reshape(-1, 1)
prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)


# In[57]:


# Compute predictions over the prediction space: y_pred
reg.fit(X, y)
y_pred = reg.predict(prediction_space)

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


# In[55]:


X


# In[8]:


s = np.array([1, 2, 3, 4, 5 ,6])


# In[19]:


s.reshape([3 , 2])


# In[11]:


s


# In[ ]:




