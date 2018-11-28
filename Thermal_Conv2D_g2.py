

#import sys, os

import sklearn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

import time, datetime




# =============================================================================
# >>> import time
# >>> print(time.time(), time.clock())
# 1359147763.02 4.95873078841e-06
# >>> time.sleep(1)
# >>> print(time.time(), time.clock())
# 1359147764.04 1.01088769662
#Both time.time() and time.clock() show that the wall-clock time
#passed approximately one second. 
#Unlike Unix, time.clock() does not return the CPU time, 
#instead it returns the wall-clock time with a higher precision than time.time().
#
# how to convert seconds to Hours:Miniutes:Seconds?
#>>> import datetime
#>>> str(datetime.timedelta(seconds=666))
#'0:11:06'
# =============================================================================

#   to clear all variables and start fresh
#   >>> clear all
#   >>> %reste

#   to display the figure inline
#   %matplotlib inline

#   to plot a figure use below example
#   plt.close
#   plt.figure
#   plt.imshow(X_train[222])

#   to wait for 5 seconds
#   time.sleep(5)

#   for checking the predictions are correct or not use below example test
#   print('\n','Y_test_predicted[99]=',Y_test_predicted[99],'\n')
#   plt.close
#   plt.figure
#   plt.imshow(X_test_original[99])


# input the data and the image dimensions
MaxSubject = 20 # Maximum number of subjects
MaxPicture = 75 # Maximu number of pictures for each subject
img_rows = 181
img_cols = 161


#############################################################

print('\n')
print('*** INITIALIZE THE VARIABLES ***')
print('\n')

X_original = np.ndarray(shape=(MaxSubject*MaxPicture,img_rows,img_cols),dtype=float)

print('\n')
print('type(X_original) = ')
print(type(X_original))
print('\n')

print('\n')
print('X_original.shape =')
print(X_original.shape)
print('\n')

Y_original = np.ndarray(shape=(MaxSubject*MaxPicture), dtype=int)

print('\n')
print('type(Y_original) = ')
print(type(Y_original))
print('\n')

print('\n')
print('Y_original.shape =')
print(Y_original.shape)
print('\n')

#from pandas import ExcelWriter
#from pandas import ExcelFile



##############################################################
Start_time_reading_format1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
Start_time_reading_format2 = time.clock()
print('\n','Start_time_reading_format1 =',Start_time_reading_format1 , '\n','Start_time_reading_format2 =',Start_time_reading_format2)
print('\n')

print('\n')
print(' *** START READING X AND Y FROM EXCEL FILE *** ')
print('\n')

Id = 0
for Subject in range(0,MaxSubject):
    print("Subject = ")
    print(Subject)
    for Picture in range(0,MaxPicture):
        FileName = 'C:/Data_Sets/Thermal_Data_Set/Copy_ThermalDataXLSX_Resized181x161/'+ str(Subject+1) + '/' + str(Picture+1) + '.xlsx'
        ### print(FileName)
        ### print(Id)
        df = pd.read_excel(FileName,sheetname='Sheet1',header=None)
        X_original[Id] = df
        Y_original[Id] = Subject 
        Id = Id + 1
        
print('\n')
print("*** END OF READING FROM EXCEL***")
print('\n')

End_time_reading_format1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
End_time_reading_format2 = time.clock()
print('\n','End_time_reading_format1 =',End_time_reading_format1 , '\n','End_time_reading_format2 =',End_time_reading_format2)
      
#Duration_time_reading_format1 = End_time_reading_format1 - Start_time_reading_format1
Duration_time_reading_format2 = End_time_reading_format2 - Start_time_reading_format2
print('\n','Duration_time_reading_format2 =',Duration_time_reading_format2 , '\n')     
Duration_time_reading_format3 = str(datetime.timedelta(seconds=Duration_time_reading_format2))
print('\n', 'Duration_time_reading_format3=',Duration_time_reading_format3)
################################################################
print('\n', ' *** RESHAPE FIRST THEN RESCALE THE X DATA ***','\n')
from sklearn import preprocessing
X_reshaped = X_original.reshape((MaxSubject * MaxPicture, img_rows * img_cols))
X_rescaled= preprocessing.scale(X_reshaped)
print('\n')
print('X_rescaled =')
print(X_rescaled)
print('\n')
 
print('\n')
print('X_rescaled.mean(axis=0) =')
print(X_rescaled.mean(axis=0))
print('\n')
 
print('\n')
print('X_rescaled.std(axis=0) = ')
print(X_rescaled.std(axis=0))
print('\n')

print('\n')
print('X_rescaled.shape = ')
print(X_rescaled.shape)
print('\n')

print('\n')
print('len(X_rescaled) = ')
print(len(X_rescaled))
print('\n')
###################################################################################################

print('\n')
print(' *** SPLIT THE DATA INTO TRAINING AND TESTING GROUPS WITH SHUFFLING ***')
print('\n')

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_rescaled, Y_original, test_size=0.20, random_state=None,shuffle=True)

#time.sleep(60)
plt.close
plt.figure
plt.imshow(X_original[3])
#time.sleep(60)

from sklearn import preprocessing

from keras.utils import to_categorical

print('X_original[1000] =')
plt.close
plt.figure
plt.imshow(X_original[1000])


print('X_original[222] =')
plt.close
plt.figure
plt.imshow(X_original[222])

Start_time_processing_format1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
Start_time_processing_format2 = time.clock()
print('\n','Start_time_processing_format1 =',Start_time_processing_format1 , '\n','Start_time_processing_format2',Start_time_processing_format2)
print('\n')
 
print('\n')
print(' *** BUILD THE MODEL, COMPILE, AND FIT ***')
print('\n')

from keras import models 
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D 
from keras.layers import MaxPooling2D, Dropout, Flatten

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=sgd)

print('\n','*** Before reshaping and categorizing X_train and Y_train ***','\n')
print('\n','X_train.shape =',X_train.shape,'\n')
X_train_original = X_train.reshape(X_train.shape[0],img_rows,img_cols) # save a copy reshaped for display
print('\n','X_train_original.shape =',X_train_original.shape,'\n')
print('\n','Y_train.shape =',Y_train.shape,'\n')
X_train_reshaped = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
Y_train_categorized = keras.utils.to_categorical(Y_train, num_classes=20)
print('\n','*** After reshaping and categorizing X_train and Y_train ***','\n')
print('\n','X_train_reshaped.shape =',X_train_reshaped.shape,'\n')
print('\n','Y_train_categorized.shape =',Y_train_categorized.shape,'\n')

print('\n','*** start fitting ***','\n')
model.fit(X_train_reshaped, Y_train_categorized, batch_size=32, epochs=25)
print('\n','*** after fitting before scoring ***','\n')

print('\n','*** Before reshaping and categorizing X_test and Y_test ***','\n')
print('\n','X_test.shape =',X_test.shape,'\n')
X_test_original = X_test.reshape(X_test.shape[0],img_rows,img_cols) # save a copy reshaped for display
print('\n','X_test_original.shape =',X_test_original.shape,'\n')
print('\n','Y_test.shape =',Y_test.shape,'\n')
X_test_reshaped = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
Y_test_categorized = keras.utils.to_categorical(Y_test, num_classes=20)
print('\n','*** After reshaping and categorizing X_test and Y_test ***','\n')
print('\n','X_test_reshaped.shape =',X_test_reshaped.shape,'\n')
print('\n','Y_test_categorized.shape =',Y_test_categorized.shape,'\n')

print('\n','*** start scoring ***','\n')
score = model.evaluate(X_test_reshaped, Y_test_categorized, batch_size=32)
print('\n','*** after scoring ***','\n')

End_time_processing_format1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
End_time_processing_format2 = time.clock()
print('\n','End_time_processing_format1 =',End_time_processing_format1 , '\n','End_time_processing_format2 =',End_time_processing_format2)     
#Duration_time_processing_format1 = End_time_processing_format1 - Start_time_processing_format1
Duration_time_processing_format2 = End_time_processing_format2 - Start_time_processing_format2
print('\n','Duration_time_processing_format2 =',Duration_time_processing_format2 , '\n')     
Duration_time_processing_format3 = str(datetime.timedelta(seconds=Duration_time_processing_format2))
print('\n','Duration_time_processing_format3 =',Duration_time_processing_format3)

print('\n','*** before model summary ***','\n')
print(model.summary()) 
print('\n','*** after model summary ***','\n')

print('\n','*** before printing model layers ***','\n')
print(model.layers) 
print('\n','*** after printing model layers ***','\n')

print('\n','*** before number of model layers ***','\n')
print('len(model.layers)= ',len(model.layers)) 
print('\n','*** after number of model layers ***','\n')

print('\n','*** before Test loss ***','\n')
print('Test loss:', score[0])
print('\n','*** after Test loss ***','\n')

print('\n','*** before Test accuracy ***','\n')
print('Test accuracy:', score[1])
print('\n','*** after Test accuracy ***','\n')
Y_test_predicted = model.predict_classes(X_test_reshaped)
print('\n','len(X_test_reshaped)=',len(X_test_reshaped),'\n')
print('\n','len(Y_test_predicted)=',len(Y_test_predicted),'\n')
print('\n','Y_test_predicted =',Y_test_predicted,'\n')
