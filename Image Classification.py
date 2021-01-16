#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[2]:


import matplotlib.pyplot as plt

image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8


# In[4]:


plt.imshow(x_train[image_index], cmap='Greys')


# In[3]:


x_train.shape


# In[12]:


x_train.shape[0]


# In[13]:


# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[15]:


input_shape = (28, 28, 1)


# In[ ]:


input_shape


# In[ ]:


# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[18]:


# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# In[19]:


# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# In[20]:


# Creating a Sequential Model and adding the layers
model = Sequential()


# In[22]:


model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))


# In[23]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[24]:


model.add(Flatten()) # Flattening the 2D arrays for fully connected layers


# In[25]:


model.add(Dense(128, activation=tf.nn.relu))


# In[26]:


model.add(Dropout(0.2))


# In[28]:


model.add(Dense(10,activation=tf.nn.softmax))


# In[29]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)


# In[30]:


model.evaluate(x_test, y_test)


# In[31]:


image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')


# In[32]:


pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())


# In[ ]:





# In[ ]:





# In[ ]:




