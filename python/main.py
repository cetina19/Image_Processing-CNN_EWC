root = ".\dataRaw"

import sys
import os
from math import log
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt


def convertAndSave(array,name):
    print('Processing '+name)
    if array.shape[1]!=16: 
        assert(False)
    b=int((array.shape[0]*16)**(0.5))
    b=2**(int(log(b)/log(2))+1)
    a=int(array.shape[0]*16/b)
    array=array[:a*b//16,:]
    array=np.reshape(array,(a,b))
    im = Image.fromarray(np.uint8(array))
    im.save(root+'\\'+name+'.png', "PNG")
    return im


files=os.listdir(root)
print('files : ',files)

for counter, name in enumerate(files):
        
        if '.bytes' != name[-6:]:
            continue
        f=open(root+'/'+name)
        array=[]
        for line in f:
            xx=line.split()
            if len(xx)!=17:
                continue
            array.append([int(i,16) if i!='??' else 0 for i in xx[1:] ])
        plt.imshow(convertAndSave(np.array(array),name))
        del array
        f.close()

path_root = ".\data\malimg_paper_dataset_imgs\\"

from keras.preprocessing.image import ImageDataGenerator
batches = ImageDataGenerator().flow_from_directory(directory=path_root, target_size=(64,64), batch_size=10000)

batches.class_indices

imgs, labels = next(batches)

imgs.shape

labels.shape


def plots(ims, figsize=(20,30), rows=10, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = 10 
    for i in range(0,50):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(list(batches.class_indices.keys())[np.argmax(titles[i])], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

plots(imgs, titles = labels)

classes = batches.class_indices.keys()

perc = (sum(labels)/labels.shape[0])*100

plt.xticks(rotation='vertical')
plt.bar(classes,perc)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imgs/255.,labels, test_size=0.3)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Input, Model

from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

import tensorflow as tf
print(tf.__version__)

num_classes = 25

def malware_model():
    Malware_model = Sequential()
    Malware_model.add(Conv2D(30, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(64,64,3)))

    Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
    Malware_model.add(Conv2D(15, (3, 3), activation='relu'))
    Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
    Malware_model.add(Dropout(0.25))
    Malware_model.add(Flatten())
    Malware_model.add(Dense(128, activation='relu'))
    Malware_model.add(Dropout(0.5))
    Malware_model.add(Dense(50, activation='relu'))
    Malware_model.add(Dense(num_classes, activation='softmax'))
    Malware_model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return Malware_model


Malware_model = malware_model()

Malware_model.build(input_shape=(None,64,64,3))
Malware_model.summary()

y_train.shape

y_train_new = np.argmax(y_train, axis=1)

y_train_new

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train_new),
                                                 y=y_train_new)

class_weights = {i : class_weights[i] for i in range(25)}

class_weights

loss_fn2 = tf.keras.losses.CategoricalCrossentropy()

Malware_model.compile(optimizer='adam', loss=loss_fn2, metrics=['accuracy'])
Malware_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25,  class_weight=class_weights)

scores = Malware_model.evaluate(X_test, y_test)

print('Final CNN accuracy: ', scores[1])

import numpy as np
import pandas as pd

y_pred = Malware_model.predict_classes(X_test, verbose=0)

y_pred

y_test2 = np.argmax(y_test, axis=1)

y_test2

from sklearn import metrics
c_matrix = metrics.confusion_matrix(y_test2, y_pred)

import seaborn as sns
def confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names= batches.class_indices.keys()
confusion_matrix(c_matrix, class_names, figsize = (20,7), fontsize=14)


import numpy as np





ewc_lambda = 0.5  
fisher_matrices = []  
importance_matrices = []  


for layer in Malware_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()[0]
        input_shape = layer.input_shape[1:]
        output_shape = layer.output_shape[1:]
        num_inputs = np.prod(input_shape)
        num_outputs = np.prod(output_shape)
        inputs = np.random.rand(100, *input_shape)
        outputs = Malware_model.predict(inputs)
        fisher_matrix = np.zeros((num_outputs, num_inputs, num_inputs))
        for i in range(100):
            output = outputs[i].reshape(num_outputs, 1)
            input_vec = inputs[i].reshape(num_inputs, 1)
            fisher_matrix += np.matmul(output, input_vec.T)
        fisher_matrix /= 100
        fisher_matrices.append(fisher_matrix)


prev_task_acc = 0.9  
new_task_acc = 0.8  
for fisher_matrix in fisher_matrices:
    importance_matrix = ewc_lambda * (fisher_matrix / prev_task_acc)
    importance_matrix += (1 - ewc_lambda) * (fisher_matrix / new_task_acc)
    importance_matrices.append(importance_matrix)


regularization_terms = []
for i, layer in enumerate(Malware_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()[0]
        regularization_term = ewc_lambda * np.matmul(importance_matrices[i], (weights**2).flatten())
        regularization_terms.append(regularization_term)


def ewc_loss(y_true, y_pred):
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    ewc_loss = cross_entropy_loss
    for regularization_term in regularization_terms:
        ewc_loss += regularization_term
    return ewc_loss


Malware_model.compile(optimizer='adam',
              loss=ewc_loss,
              metrics=['accuracy'])









batches_f = ImageDataGenerator().flow_from_directory(directory=path_root, target_size=(64,64), batch_size=10000)
imgs_f, labels_f = next(batches_f)

classes_f = batches_f.class_indices.keys()

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(imgs_f/255.,labels_f, test_size=0.3)
y_train_new_f = np.argmax(y_train_f, axis=1)

loss_fn_f = tf.keras.losses.CategoricalCrossentropy()
Malware_model.fit(X_train_f, y_train_f, epochs=25, validation_data=(X_test_f, y_test_f))

scores2 = Malware_model.evaluate(X_test_f, y_test_f)

original_task_change = Malware_model.evaluate(X_test, y_test)
new_task_change = Malware_model.evaluate(X_test_f, y_test_f)

original_task_change = scores[0] - original_task_change[0], scores[1] - original_task_change[1]
new_task_change = scores2[0] - new_task_change[0], scores2[1] - new_task_change[1]

print("First = ",scores," *** Second = ",scores2)
print("Original Task Change = ",original_task_change,"\nNew Task Change = ",new_task_change)

if(original_task_change[0]>=0 and original_task_change[1]<=0):
    print("Previous Task accuracy and loss didn't effected negatively")
if(new_task_change[0]>=0 and new_task_change[1]<=0):
    print("New Task accuracy and loss didn't effected negatively")