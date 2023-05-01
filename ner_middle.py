from flask import Flask, request
import requests
import pandas as pd
import os
import numpy as np
from pprint import pprint
import paramiko
import json
from pprint import pprint
app = Flask(__name__)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# connect to the server
ssh.connect(hostname="10.10.10.165", username="nuhhatipoglu", password="AAaa321321..,")
# create new pandas dataframe with annotations

request_count = 0
limit =  3


def convert_annotations(json_data):
    text = json_data["task"]["data"]["text"]
    annotations = json_data["annotation"]["result"]

    labeled_text = []
    prev_end = 0
    for a in annotations:
        start = a["value"]["start"]
        end = a["value"]["end"]
        label = a["value"]["labels"][0]

        # Add non-labeled text between labels (including whitespaces)
        if start > prev_end:
            non_labeled_text = text[prev_end:start].split()
            for word in non_labeled_text:
                labeled_text.append((word, "O"))

        # Add labeled text
        labeled_text.append((a["value"]["text"], label))
        prev_end = end

    # Add any remaining non-labeled text after the last label
    if prev_end < len(text):
        non_labeled_text = text[prev_end:].split()
        for word in non_labeled_text:
            labeled_text.append((word, "O"))

    return labeled_text

#text = "Ankara, Türkiye'nin başkenti ve en büyük ikinci şehri olarak büyümeye devam ediyor."

#converted_labels = convert_labels(result, text)

#for token, label in converted_labels:
#    print(token, label)

labeled_data = []
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

    labeled_text = convert_annotations(request.get_json())
    
    # Open a new text file in write mode
    labeled_data.append(labeled_text)

    
    if request_count == limit:
        # open sftp connection
        sftp = ssh.open_sftp()

        with open('labeled_text.txt', 'w', encoding='utf-8') as f:
        # Loop through each item in the list
            for labeled_text in labeled_data:
                for word, label in labeled_text:
                    # Write the word and label to the file, separated by a space and followed by a newline character
                    f.write(word + ' ' + label + '\n')
        
        labeled_data.clear()
        # copy the numpy files to the server
        sftp.put("labeled_text.txt", "/home/nuhhatipoglu/SAnnotate-NLP-v2/data/DataSet-1/" + "labeled_text.txt")
        sftp.close()
        # make post request to trigger the next task to remote server
        requests.post("http://10.10.10.165:5000/", data = {'MESSAGE': 'OKKK!'})
        print("=========Numpy files copied to remote folder=======")
        request_count = 0
        # close the sftp connection
        

    return "Success"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


