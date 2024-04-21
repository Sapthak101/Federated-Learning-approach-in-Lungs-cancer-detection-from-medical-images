import flwr as fl
import numpy as np
import sys
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, TimeDistributed, Bidirectional, LSTM, GRU, Dense, Dropout, Input, concatenate
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Reshape
import os
import cv2
from keras.regularizers import l2
import argparse
import os
from catboost import CatBoostClassifier, Pool
import warnings 
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

#Data Preprocessing
def preprocessing_denoise(img):
    denoise_img = cv2.medianBlur(img, 1)
    denoise_img = cv2.cvtColor(denoise_img, cv2.COLOR_BGR2RGB)
    return denoise_img

df = pd.read_csv('Dataset 2\Client2_Data.csv')

df_train, df_test = train_test_split(df, train_size = 0.70, random_state = 42)

df_train,df_val=train_test_split(df, train_size = 0.70, random_state = 42)

df_train=pd.DataFrame(df_train)
df_val=pd.DataFrame(df_val)
df_test=pd.DataFrame(df_test)

IMG_WIDTH = 224
IMG_HEIGHT = 224

image_size = (IMG_WIDTH, IMG_HEIGHT)
batch_size = 32

TRAIN_DATAGEN = ImageDataGenerator(rescale = 1./255.,
                                   preprocessing_function = preprocessing_denoise,
                                  rotation_range = 30,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.2,
                                  shear_range = 0.1,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

TEST_DATAGEN = ImageDataGenerator(rescale = 1./255.)

train_generator = TRAIN_DATAGEN.flow_from_dataframe(
    df_train,
    x_col = "image_path",
    y_col = "image_label",
    target_size = image_size,
    batch_size = batch_size,
    color_mode = "rgb",
    class_mode = "categorical",
    shuffle = True
)
                
val_generator = TEST_DATAGEN.flow_from_dataframe(
    df_val,
    x_col = "image_path",
    y_col = "image_label",
    target_size = image_size,
    batch_size = batch_size,
    color_mode = "rgb",
    class_mode = "categorical",
    shuffle = True
)

test_generator = TEST_DATAGEN.flow_from_dataframe(
    df_test,
    x_col = "image_path",
    y_col = "image_label",
    target_size = image_size,
    batch_size = batch_size,
    color_mode = "rgb",
    class_mode = "categorical",
    shuffle = True
)

classes = list(train_generator.class_indices.keys())

#Weight Class
class_weights = class_weight.compute_class_weight(
           class_weight = 'balanced',
            classes = np.unique(train_generator.classes), 
            y = train_generator.classes)

train_class_weights = dict(enumerate(class_weights))

for idx, weight, in train_class_weights.items():
    class_name = classes[idx]
    print(f"{class_name} : {weight}")

pre_trained_model = InceptionV3(input_shape = (IMG_WIDTH, IMG_HEIGHT, 3), 
                                include_top = False, 
                                weights = "imagenet")

for layer in pre_trained_model.layers:
     layer.trainable = True

model = Sequential()
model.add(Input(shape = (IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(pre_trained_model)
model.add(Flatten())
model.add(Dense(1024, activation = "relu",kernel_regularizer=l2(0.01)))
model.add(Dense(512, activation = "relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(3, activation = "softmax"))

print(model.summary())

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

from tqdm import tqdm
i=0
test_generator.reset()
X_test, y_test = next(test_generator)
for i in tqdm(range(int(test_generator.n/batch_size)-1)): 
    img, label = next(test_generator)
    X_test = np.append(X_test, img, axis=0 )
    y_test = np.append(y_test, label, axis=0)
#X_test=X_test.reshape(len(X_test), 30000)


class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.bst = None
        self.config = None
        self.num=20
        self.index=0
        self.index1=0
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        global hist
        hist=[]

        model.set_weights(parameters)
        epochs = 50
        history = model.fit(train_generator,
                   steps_per_epoch = len(train_generator),
                   batch_size = 32,
                   validation_data = val_generator,
                   validation_steps = len(val_generator),
                   callbacks=[
                               EarlyStopping(monitor = "val_loss",
                               patience = 5,
                               restore_best_weights = True), 
                               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, mode='min') 
                              ],
                   epochs = epochs)
        
        model_loss=pd.DataFrame(model.history.history)
        model_loss.plot()
        plt.xlabel('Number of Epochs')
        str1="Loss_accu_image_fed_round_client2"+str(self.index1)+".png"
        plt.savefig(str1)

        self.index1=self.index1+1
        #print("Fit history : " ,hist)
        return model.get_weights(), 383, {}
    
    def evaluate(self, parameters, config):
        global y_pred
        model.set_weights(parameters)
        loss, accuracy=model.evaluate(test_generator)
        y_pred=model.predict(X_test)
        print("Eval accuracy : ", accuracy)
        return loss, 165, {"accuracy": accuracy}
    
fl.client.start_client(
    server_address="127.0.0.1:16080", 
    client=CifarClient().to_client())

#Perform the required analysis
probabilities = y_pred

indices = np.argmax(probabilities, axis=1)

one_hot_probabilities = np.zeros((probabilities.shape[0], probabilities.shape[1]))
one_hot_probabilities[np.arange(probabilities.shape[0]), indices] = 1

labels=['0: Normal', '1: Malignant', '2: Benign']
print(labels)

y_pred=one_hot_probabilities

kappa = cohen_kappa_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Cohen's kappa score:", kappa)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
plt.figure(figsize=(20,10))
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax); 

labels=['Benign', 'Malignant', 'Normal']
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

plt.savefig("confusion_matrix_client2.png")

print(classification_report(y_test, y_pred, labels=[0,1,2]))