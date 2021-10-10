import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy

from keras.regularizers import l2
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

df=pd.read_csv('fer2013.csv')


X_train,train_y,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")


num_labels = 7
batch_size = 64
epochs = 15
width, height = 48, 48


X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

#normalizacija vrednosti izmedju 0 i 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)



# %% kreiranje konvolucione neur. mreze
#prvi konvolucioni sloj

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides=(2, 2)))
model.add(Dropout(0.4))
    
#drugi konvolucioni sloj
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides=(2, 2)))
model.add(Dropout(0.4))

#treci konvolucioni sloj
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides=(2, 2)))
model.add(Dropout(0.4))

#cetvrti konvolucioni sloj
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides=(2, 2)))
model.add(Dropout(0.4))
    
model.add(Flatten())

#potpuno povezana konv. mreza
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(num_labels, activation='softmax'))

# model.summary()

#kompajliranje modela
model.compile(loss=categorical_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])

print(model.summary())


#treniranje modela
history = model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)




#cuvanje modela radi ponovnog koriscenja
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")


