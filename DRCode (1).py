#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

trainLabels = pd.read_csv(r'C:\Users\User\Desktop\PROJECTS\ug\input3\trainLabels.csv')
trainLabels.head()


# In[5]:


from PIL import Image
from keras.preprocessing import image
import os
import numpy as np

# resize the image to (256, 256)
img_rows, img_cols = 256, 256

listing = os.listdir(r'C:\Users\User\Desktop\PROJECTS\ug\input3') 
listing.remove('trainLabels.csv')

immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename(r'C:\Users\User\Desktop\PROJECTS\ug\input3' + file)
    fileName = os.path.splitext(base)[0]
    fileName1=fileName.replace("input3","")
    print(fileName1)
    print(trainLabels.image==fileName1)
    print(trainLabels.loc[trainLabels.image==fileName1, 'level'])
    #print(trainLabels.loc['10_left', 'level'])
    imlabel.append(trainLabels.loc[trainLabels.image==fileName1, 'level'].values[0])
    im = Image.open(r'C:/Users/User/Desktop/PROJECTS/ug/input3/' + file)
    img = np.array(im.resize((img_rows,img_cols)))
    
    # convert to green channel only
    img[:,:,[0,2]] = 0
    immatrix.append(img)
    print("IMLABEL")
    print(imlabel)
    print("IMMATRIX")
    print(immatrix)


# In[6]:


im = Image.fromarray(immatrix[1],'RGB')
print("level:",imlabel[1])
im


# In[8]:


import random

# define transformation methods
def horizontal_flip(image_array):
    return image_array[:, ::-1]

def vertical_flip(image_array):
    return image_array[::-1,:]

def random_transform(image_array):
    if random.random() < 0.5:
        return vertical_flip(image_array)
    else:
        return horizontal_flip(image_array)


# In[9]:


im = Image.fromarray(vertical_flip(immatrix[1]),'RGB')
im


# In[10]:


length = len(immatrix)
for i in range(length):
    if random.random() < 0.1:
        immatrix.append(random_transform(immatrix[i]))
        imlabel.append(imlabel[i])
        
print("Size of image array before augmentation: ", length)
print("Size fo image array after augmentation: ", len(immatrix))


# In[11]:


from sklearn.utils import shuffle

data,label = shuffle(immatrix, imlabel, random_state=42)
train_data = [data,label]
print(train_data)


# In[12]:


import matplotlib.pyplot as plt

plt.hist(label)
plt.xlabel('DR Range')
plt.ylabel('Number of patients')
plt.show()


# In[13]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size = 0.1, random_state = 42)

print(np.array(x_train).shape)
print(np.array(y_train).shape)


# In[14]:


from keras.utils import np_utils

y_train = np_utils.to_categorical(np.array(y_train), 5)
y_test = np_utils.to_categorical(np.array(y_test), 5)

x_train = np.array(x_train).astype("float32")/255.
x_test = np.array(x_test).astype("float32")/255.

print(np.array(y_train).shape)


# In[18]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train[0].shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(activation='softmax', units=5))

model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# In[19]:


model.summary()


# In[20]:


model.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2)


# In[24]:


predictions = model.predict(x_test)
print(predictions)
print(y_test)


# In[25]:


score = model.evaluate(x_test, y_test, verbose=0)
print(score[1])


# In[ ]:




