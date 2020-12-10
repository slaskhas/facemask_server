#    "git clone https://github.com/prajnasb/observations"

import os
import numpy as np
import matplotlib.pyplot as plt


from keras.layers import Dense, GlobalAveragePooling2D

from keras.models import Sequential, load_model, Model
from keras.applications.mobilenet import MobileNet


from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.python.keras import backend as K
sess = tf.compat.v1.Session()
K.set_session(sess)

model = Sequential()
mobilenet = MobileNet()

for layer in range(len(mobilenet.layers)-1):
    model.add(mobilenet.layers[layer])

for layer in model.layers[:-9]:
    layer.trainable = False

model.add(Dense(2, activation = 'softmax'))

model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

mobilenet = MobileNet(include_top = False)
x = mobilenet.output

x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)

preds = Dense(2,activation='softmax')(x)

model = Model(inputs = mobilenet.input, outputs=preds)

for layer in model.layers[:-9]:
    layer.trainable = False

model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.save('untrained_mdl.h5')
mdl = load_model('untrained_mdl.h5')

model.summary()

for l in model.layers:
    print(l,'\t\t',l.trainable)

# mobilenet.summary()

########

os.listdir('observations/experiements/data/with_mask')

X = []
y = []
for i in os.listdir('observations/experiements/data/with_mask'):
    img = load_img('observations/experiements/data/with_mask/'+str(i), target_size = (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    X.append(img)
    y.append([1,0])

for i in os.listdir('observations/experiements/data/without_mask'):
    img = load_img('observations/experiements/data/without_mask/'+str(i), target_size = (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    X.append(img)
    y.append([0,1])

for i in os.listdir('my_data/with_mask/'):
    img = load_img('my_data/with_mask/'+str(i), target_size = (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    X.append(img)
    y.append([1,0])

for i in os.listdir('my_data/without_mask'):
    img = load_img('my_data/without_mask/'+str(i), target_size = (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    X.append(img)
    y.append([0,1])

data = np.array(X)
labels = np.array(y)

(trainX, testX, trainY, testY) = train_test_split(data, labels)

model.compile(loss="binary_crossentropy", optimizer = 'adam', metrics=["accuracy"])

H = model.fit( trainX, trainY, epochs = 25, batch_size = 32, validation_split = 0.1)

model.evaluate(testX , testY)

img = load_img('observations/experiements/data/without_mask/376.jpg', target_size = (224,224))
img = img_to_array(img)
img = preprocess_input(img)
#plt.imshow(img)

img = np.expand_dims(img, axis = 0)
model.predict(img)

img = load_img('./mask.png', target_size = (224,224))
img = img_to_array(img)
img = preprocess_input(img)
#plt.imshow(img)

img = np.expand_dims(img, axis = 0)

model.predict(img)

builder = tf.compat.v1.saved_model.Builder("masknet")
builder.add_meta_graph_and_variables(sess, ["msk"])

#model.save('MaskNetV2.hdf5')

builder.save()
sess.close()
