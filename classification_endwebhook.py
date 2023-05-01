import argparse
import os
import json
import gc
import shutil
import numpy as np
import requests
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
mirrored_strategy = tf.distribute.MirroredStrategy()
import matplotlib.pyplot as plt
import sys
import logging
import warnings
from flask import Flask, request
warnings.filterwarnings('ignore')
app = Flask(__name__)


SAVE_PATH = "C:\\Project Sannotate\\JELAL\\classification"

# TODO: 1) change dataset size returned in initialize_dataset_v2() function =================== DONE
#       2) change model in VGG16_Model_Type3() function ======================================= DONE
#       3) test model with unnannotated dataset and check entropy and indexes of images =======
#       4) test the Active learning pipeline with Label Studio 

def delete_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}: {e}')

def initialize_dataset_v2():
    path = "classification/"

    x_initial = np.load(path + "train_images.npy")
    y_initial = np.load(path + "train_labels.npy")
    
    #x_initial = x_initial.astype('float32')
    #y_initial = to_categorical(y_initial, 3)
    n_classes = y_initial.shape[1]
    #n_classes =int(y_initial.max() + 1)
    x_test = np.load(path + "2500_x_test.npy")
    y_test = np.load(path + "2500_y_test.npy")

    x_valid = np.load(path + "2500_x_valid.npy")
    y_valid = np.load(path + "2500_y_valid.npy")
    
    return x_initial, y_initial, x_test[0:1500], y_test[0:1500], x_valid[0:1700], y_valid[0:1700], n_classes



# get unannotated images from "unannotated images" folder and convert them to numpy array using create_dataset() function
def unannotated_dataset(path):
    """
    This function takes path to unannotated images (from Label Studio) folder 
    and returns numpy array of unannotated images and their labels
    :param path: path to unannotated images folder
    :return: numpy array of unannotated images and their labels
    """
    data = []
    labels = []
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img = plt.imread(img_path)
        data.append(img)
        # get label from image name: For example image name is 451_0ed6edd1-label_number_1_0_658. And label is right after label_number_ and before _ 
        label = int(img_path.split('-')[-1].split('_')[0]) # TODO: need to change this line to get labels from image names
        labels.append(label)
    return np.array(data), np.array(labels)

class CustomCallback(K.callbacks.Callback):
    def on_train_end(self, epoch, logs=None):
        logs = logs or {}
        keys = list(logs.keys())
        accuracy = logs.get("accuracy")
        loss = logs.get("loss")
        print("Finished training == got log keys: {}".format(keys))
        with open("training_metrics.txt","a") as f:
            f.write(f"Epoch {epoch}: accuracy = {accuracy}, loss ={loss}\n")


def initialize_model(x_train, y_train, x_valid, y_valid, number_of_classes, data_input_shape):

    """
    model = load_model('classification/Tuborg54_Initial_2500_CEAL_VGG_Type3_en_v2.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=1),
        ModelCheckpoint("classification/Tuborg54_Initial_2500_CEAL_VGG_Type3_en_v2.h5",
                        monitor='val_acc', save_best_only=True)
    ]

    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              batch_size=8,
              epochs=20,
              callbacks=callbacks)
    """
    """    
    vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=data_input_shape)

    vgg_base.trainable = True
    set_trainable = False

    for layer in vgg_base.layers:
        if layer.name == "block3_conv1":
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    x = layers.Flatten()(vgg_base.output)
    dropout1 = layers.Dropout(rate=0.5)(x)
    dense = layers.Dense(units=256, activation="relu")(dropout1)
    dropout2 = layers.Dropout(rate=0.5)(dense)
    prediction = layers.Dense(units=number_of_classes, activation="softmax")(dropout2)

    model = models.Model(inputs=vgg_base.input, outputs=prediction)

    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.RMSprop(lr=1e-5), metrics=["acc"])
    """
    model = load_model("classification/VGG_CELAL.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=1),
        ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-5),
        CustomCallback()
    ]
    
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              batch_size=8,
              shuffle=False,
              epochs=20,
              verbose=1,
              callbacks=callbacks)
    model.save("classification/VGG_CELAL.h5")
    return model



# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    # entropy = stats.entropy(y_pred_prob.T)
    # entropy = np.nan_to_num(entropy)
    origin_index = np.arange(0, len(y_pred_prob))
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)
    eni = np.column_stack((origin_index,
                           entropy,
                           pred_label))

    eni = eni[(-eni[:, 1]).argsort()]
    #print("ENTROPIES",eni[:n_samples], eni[:, 0].astype(int)[:n_samples])
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]


@app.route("/", methods=["POST", "GET"])
def predict():
    
    #request.get_json(force=True, silent=True)
   
    # get dataset from initialize_dataset_v2() function
    x_initial, y_initial, x_test, y_test, x_valid, y_valid, n_classes = initialize_dataset_v2()
    print(n_classes)
    print("=====Dataset loaded=======")
    # train model with received annotated data from Label Studio
    # log INFO to console about training process
    model = initialize_model(x_initial, y_initial, x_valid, y_valid, n_classes, x_initial[-1].shape)
    print("=====Model trained=======")
    # make POST request to Label Studio to get unannotated images
    requests.post("http://10.10.10.165:8000/copyfiles", json={"data":"1"})
    # convert unannotated images to numpy array using unannotated_dataset() function
    x_pool, y_pool = unannotated_dataset("classification_un_images")
 
    print("====Unannotated images loaded====")
    # make predictions on the unlabeled data from "unannotated images" folder, find their entropy and rank them and select the top n_samples and show corresponding image's index
    y_pred_prob = model.predict(x_pool)
    print("=====Predictions made=====")
    enis, indexes = entropy(y_pred_prob, n_samples=10)
    """
    print(y_pred_prob[0])
    print(y_pred_prob.shape)
    print(tf.nn.softmax(y_pred_prob[0][0]))
    print(np.argmax(y_pred_prob[0], axis=-1))
    """
    folder = "classification_un_images/"
    image_paths = []
    # get the list of images' paths
    for filename in os.listdir(folder):
        if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(folder, filename))
    result_dict = {}
    for index in indexes:
        print("INDEX NUMBER:  ",index)
        image_name = image_paths[int(index)]
        print("IMAGE NAME: ",image_name)
        task_id = int(image_name.split("/")[1].split("_")[0])
        answer = np.argmax(y_pred_prob[index], axis=-1)
        payload_dict = {
                "task": task_id,
                "result": int(answer),
                "score": float(100 * np.max(tf.nn.softmax(y_pred_prob[index])))
                }
        
        requests.post(url="http://10.10.10.165:8000/receive", json=payload_dict)
    # save model to h5 after training with received annotated data from Label Studio
    delete_files("classification_un_images")
    print("DELETED IMAGES")
    tf.keras.backend.clear_session()
    gc.collect()
    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)