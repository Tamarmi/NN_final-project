
import scipy.io
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# loading the data from matlab
data_correl = scipy.io.loadmat('C:/Users/ASUS/Desktop/NN_project/data_correl.mat')
mat = data_correl['dataCorrel']

list_allSubjMat = []
list_allLabels = []
# creating a list of all correlation matrices (15 healthy + 15 MCS + 11 VS) and
# a list off the corresponding lables (0-healthy 1-MCS 2-VS)
for i in range(len(mat[0])):
    for j in range(len(mat[0][i])):
        list_allSubjMat.append(mat[0][i][j][0])
        list_allLabels.append(i)

# converting the input data to vectors
array_allSub = np.asarray(list_allSubjMat)
allSub_allConnVec = array_allSub.reshape(41,121)
allSub_allConnVec = allSub_allConnVec.astype('float32')

# converting the input labels to categorical (3 categories)
array_allLables = np.asarray(list_allLabels)
allSub_categorLables = to_categorical(array_allLables)


# splitting the data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(allSub_allConnVec, allSub_categorLables, test_size=0.5)

classifier = Sequential()

classifier.add(Dense(32, activation='tanh', kernel_initializer='random_normal', input_shape=(121,)))
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(3, activation='softmax', kernel_initializer='random_normal'))

# import pydot
# import graphviz
# plot_model(classifier, show_shapes=True,show_layer_names=True)

classifier.compile(optimizer = Adam(learning_rate =.0001),loss='categorical_crossentropy', metrics =['categorical_accuracy'])

historyTrain=classifier.fit(X_train,y_train, batch_size=32, epochs=1000, shuffle=True, validation_data=(X_test,y_test))


loss_train = historyTrain.history['loss']
loss_test = historyTrain.history['val_loss']

epochs = range(1000)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_test, 'b', label='test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accuracy_train = historyTrain.history['categorical_accuracy']
accuracy_test = historyTrain.history['val_categorical_accuracy']

plt.plot(epochs, accuracy_train, 'g', label='Training accuracy')
plt.plot(epochs, accuracy_test, 'b', label='Test accuracy')
plt.title('Training and Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




