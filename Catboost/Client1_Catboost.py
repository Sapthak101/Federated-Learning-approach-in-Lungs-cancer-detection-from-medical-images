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
from catboost import CatBoostClassifier, Pool, cv

def preprocessing_denoise(img):
    denoise_img = cv2.medianBlur(img, 1)
    denoise_img = cv2.cvtColor(denoise_img, cv2.COLOR_BGR2RGB)
    return denoise_img

df = pd.read_csv('Dataset 2\Client1_Data.csv')

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

from tqdm import tqdm
j=0
classes = list(train_generator.class_indices.keys())

class_weights = class_weight.compute_class_weight(
           class_weight = 'balanced',
            classes = np.unique(train_generator.classes), 
            y = train_generator.classes)

train_class_weights = dict(enumerate(class_weights))

for idx, weight, in train_class_weights.items():
    class_name = classes[idx]
    print(f"{class_name} : {weight}")
    j=0

i=0
train_generator.reset()
X_train, y_train = next(train_generator)
for i in tqdm(range(int(train_generator.n/batch_size)-1)): 
    img, label = next(train_generator)
    X_train = np.append(X_train, img, axis=0 )
    y_train = np.append(y_train, label, axis=0)
print(X_train.shape, y_train.shape)
X_train=X_train.reshape(len(X_train), 150528)

i=0
val_generator.reset()
X_val, y_val = next(val_generator)
for i in tqdm(range(int(val_generator.n/batch_size)-1)): 
    img, label = next(val_generator)
    X_val = np.append(X_val, img, axis=0 )
    y_val = np.append(y_val, label, axis=0)
X_val=X_val.reshape(len(X_val), 150528)

i=0
test_generator.reset()
X_test, y_test = next(test_generator)
for i in tqdm(range(int(test_generator.n/batch_size)-1)): 
    img, label = next(test_generator)
    X_test = np.append(X_test, img, axis=0 )
    y_test = np.append(y_test, label, axis=0)
X_test=X_test.reshape(len(X_test), 150528)


X_train=X_train
y_train=y_train
X_val=X_val
y_val=y_val
X_test=X_test
y_test=y_test

model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=2, loss_function='MultiLogloss') 
train_pool=Pool(data=X_train, label=y_train)
eval_pool=Pool(data=X_val, label=y_val)

class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.bst = None
        self.config = None
        self.index1=0

    def get_parameters(self, config):
        return train_pool.get_weight()
        
    def fit(self, parameters, config):
        global history
        
        train_pool=Pool(data=X_train, label=y_train, weight=np.array(parameters))

        if (self.bst==None):
            print("Starting of local Round 1st: ")
            history=model.fit(X=train_pool, verbose=True, early_stopping_rounds=10)
            print("Ending of local Round 1st: ")
            self.bst=model
            metrics=model.eval_metrics(eval_pool,
                             metrics=['MultiLogloss'], 
                             plot = True)            
            str1="Loss_image_fed_round_client1"+str(self.index1)+".png"
            plt.title('Catboost Loss')
            plt.savefig(str1)

            y_pred_val=model.predict(X_val)
            loss=log_loss(y_val, y_pred_val)
            print("val loss: ", loss)
            self.index1=self.index1+1
            return train_pool.get_weight(), len(X_train), {}
        else:
            print("Starting of local Round: ")
            history=model.fit(X=train_pool, verbose=True, init_model=self.bst, early_stopping_rounds=10)
            self.bst=model
            print("Ending of local Round: ")
            metrics=model.eval_metrics(eval_pool,
                             metrics=['MultiLogloss'], 
                             plot = True)
            
            str1="Loss_image_fed_round_client1"+str(self.index1)+".png"
            plt.title('Catboost Loss')
            plt.savefig(str1)

            y_pred_val=model.predict(X_val)
            loss=log_loss(y_val, y_pred_val)
            print("val loss: ", loss)
            self.index1=self.index1+1
            return train_pool.get_weight(), len(X_train), {}

    def evaluate(self, parameters, config):
        model=self.bst
        global y_pred
        train_pool=Pool(data=X_train, label=y_train, weight=np.array(parameters))

        model = CatBoostClassifier(iterations=10, learning_rate=0.1, depth=2, loss_function='MultiLogloss') 
        model.fit(X=train_pool, init_model=self.bst, early_stopping_rounds=10)
        self.bst=model

        y_pred=model.predict(X_test)
        accuracy=accuracy_score(y_test, y_pred)
        loss=log_loss(y_test, y_pred)
        print("Test accuracy : ", accuracy)

        return loss, len(X_test), {"accuracy": accuracy}

fl.client.start_client(
    server_address="127.0.0.1:18080", 
    client=CifarClient().to_client())

#Perform the required analysis
labels=['0: Malignant', '1: Benign', '2: Normal']
print(labels)

kappa = cohen_kappa_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Cohen's kappa score:", kappa)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
plt.figure(figsize=(20,10))
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax); 

labels=['Malignant', 'Benign', 'Normal']
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

plt.savefig("confusion_matrix_client1.png")

print(classification_report(y_test, y_pred, labels=[0,1,2]))