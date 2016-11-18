from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
import csv
import sys
import pickle
import numpy as np



model = load_model(sys.argv[2])
test = pickle.load( open( sys.argv[1]+'test.p', "rb" ))
test_data = np.array(test['data'])
X_test = test_data.reshape(10000,3,32,32).transpose(0,2,3,1)
X_test = X_test.astype('float32')
X_test /= 255
predict = model.predict_proba(X_test)
Class = np.argmax(predict, axis=1)
with open(sys.argv[3],'w') as f:
    w = csv.writer(f)
    w.writerows([['ID','class']])
    for i in range(len(Class)):
        w.writerows([[i,Class[i]]])