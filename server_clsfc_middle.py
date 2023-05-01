from flask import Flask, request
import requests
import pandas as pd
import os
import numpy as np
from pprint import pprint
import paramiko
import shutil
from create_data import read_images_and_labels
app = Flask(__name__)

# create new pandas dataframe with annotations

request_count = 0
limit =  15
images = []
classes = []

@app.route("/",methods=["POST", "GET"])
def receive_webhook():


    """
    This function receives the webhook from Label Studio and saves the annotations to a csv file to be used for training the model 
    and also triggers the next task if the number of annotations reaches 10.
    """
    
    
    global request_count
    
    request_count +=1

    
    print("Request count: ", request_count)

    # get the number of annotations
    image_class = request.get_json()["annotation"]["result"][0]["value"]["choices"][0]
    # CHANGE THE PATH IMAGES
    image_path = "/home/jsultanov/.local/share/label-studio/media/upload/1/" + request.get_json()["task"]["data"]["image"].split("/")[-1]
    images.append(image_path)
    classes.append(image_class)

    
    if request_count == limit:
        
        images_npy, labels_npy = read_images_and_labels(images,classes)
        
        np.save("train_images.npy", images_npy)
        np.save("train_labels.npy", labels_npy)

        # clear the lists for the next task
        images.clear()
        classes.clear()

        # copy the numpy files to the server
        source_file1 = "train_images.npy"
        destination_file1 = "classification/train_images.npy"
        shutil.copy(source_file1, destination_file1)

        source_file2 = "train_labels.npy"
        destination_file2 = "classification/train_labels.npy"
        shutil.copy(source_file2, destination_file2)
        # make post request to trigger the next task to remote server
        requests.post("http://127.0.0.1:5000/", data = {'MESSAGE': 'OK'})
        print("=========Numpy files copied to remote folder=======")
        request_count = 0
    
        

    return "Success"



if __name__ == '__main__':
    app.run(host='0.0.0.0')
