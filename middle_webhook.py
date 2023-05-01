from flask import Flask, request
import requests
import pandas as pd
import os
from pprint import pprint
import paramiko
app = Flask(__name__)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# connect to the server
ssh.connect(hostname="10.10.10.165", username="jsultanov", password="Bil444islem..")
# create new pandas dataframe with annotations


if not os.path.exists("annotations.csv"):
    df = pd.DataFrame(columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
else:
    df = pd.read_csv("annotations.csv", error_bad_lines=False)


request_count = 0
limit =  10
annotations = []


@app.route("/",methods=["POST"])
def receive_webhook(df=df):
    """
    This function receives the webhook from Label Studio and saves the annotations to a csv file to be used for training the model 
    and also triggers the next task if the number of annotations reaches 10.
    """
    
    
    global request_count
    
    request_count +=1

    # open sftp connection
    sftp = ssh.open_sftp()


    # create an empty list to store the annotations
    objects = []


    # get the number of annotations
    num_annotations = len(request.get_json()["annotation"]["result"])

    
    # print(num_annotations)
    # loop through the annotations
    for i in range(0,num_annotations):
        
        width,height, x,y = (request.get_json()["annotation"]["result"][i]["value"][k] for k in ("width", "height", "x", "y"))
        
        # transform Label Studio annotation coordinates to true pixel coordinates. 
        # Beacuse FasterRCNN requires true pixel coordinates.
        # Label Studio annotation style - The units the x, y, width and height of image annotations are provided in percentages of overall image dimension.

        pixel_x = x / 100.0 * request.get_json()["annotation"]["result"][i]["original_width"]
        pixel_y = y / 100.0 * request.get_json()["annotation"]["result"][i]["original_height"]
        pixel_width = width / 100.0 * request.get_json()["annotation"]["result"][i]["original_width"]
        pixel_height = height / 100.0 * request.get_json()["annotation"]["result"][i]["original_height"]

        path = request.get_json()["task"]["data"]["image"]
        labels = request.get_json()["annotation"]["result"][i]["value"]["rectanglelabels"][0]
        objects.append({"filename": path.split("/")[-1], "width": request.get_json()["annotation"]["result"][i]["original_width"], "height": request.get_json()["annotation"]["result"][i]["original_height"], "class": labels, "xmin": int(pixel_x),"ymin": int(pixel_y),  "xmax": int(pixel_x + pixel_width), "ymax": int(pixel_y + pixel_height)})
        # append the annotations to the dataframe
        annotations.append([path.split("/")[-1], request.get_json()["annotation"]["result"][i]["original_width"], request.get_json()["annotation"]["result"][i]["original_height"], labels, int(pixel_x),  int(pixel_y), int(pixel_x + pixel_width),  int(pixel_y + pixel_height)])
    df = df.append(objects, ignore_index = False)
            
    # save the dataframe to a csv file
    #df.to_csv("annotations.csv", mode="a", index=False, header = not os.path.exists("annotations.csv"))
    print("{}'th POST REQUEST RECEIVED".format(request_count))





    # check if created pandas datafram file in local directory contains 10 annotations and if it is reached then trigger the next task
    if request_count == limit:

        # rows = len(annotations)

        #num_rows_70_percent = rows // 10 * 7
        #testing_annotations = annotations[num_rows_70_percent:]
        training_annotations = annotations
        
        # get image names from annotations list for training and testing data splits
        tr_image_names = [image[0] for image in training_annotations]
        #ts_image_names = [image[0] for image in testing_annotations]

        # unique image names for training and testing
        tr_unique_imgs = list(set(tr_image_names))
        #ts_unique_imgs = list(set(ts_image_names) - set(tr_unique_imgs))

        print("TRAINING IMAGES: ",len(tr_unique_imgs))
        #print("TESTING IMAGES: ",len(ts_unique_imgs))

        for tr_image in tr_unique_imgs:

            print("TRAINING COPYING IMAGE for  ",tr_image)
            image_path = "C:\\Users\\jelal.sultanov\\AppData\\Local\\label-studio\\label-studio\\media\\upload\\18\\" + tr_image
            sftp.put(image_path, "train_media/" + tr_image)
        """    
        for ts_image in ts_unique_imgs:
            
            print("TESTING COPYING IMAGE for  ",ts_image)
            image_path = "C:\\Users\\jelal.sultanov\\AppData\\Local\\label-studio\\label-studio\\media\\upload\\18\\" + ts_image
            sftp.put(image_path, "test_media/" + ts_image)
        """
        print("Images uploaded successfully to train and test folders in server")
        request_count = 0
        # take last X rows from csv file and copy it 
        # select the last 20 rows of the DataFrame
        train_df = pd.DataFrame(training_annotations, columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
        # test_df = pd.DataFrame(testing_annotations, columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
        # merge train_df and test_df on all columns

        # merge dataframes on 'filename'
        #merged_df = test_df.merge(train_df[['filename']], on='filename', how='left', indicator=True)

        # drop rows where '_merge' is 'both'
        #test_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)


        # save the new DataFrame to a CSV file
        train_df.to_csv('train.csv', index=False)
        # test_df.to_csv("test.csv", index=False)
        # print("Triggering the next task")
        sftp.put('train.csv', "train_media/" + "train.csv")
        #sftp.put('test.csv', "test_media/" + "test.csv")

        print("=========CSV FILES  train.csv AND test.csv COPIED TO REMOTE FOLDER=======")
        #os.remove("new_file.csv")
        annotations.clear()
        tr_unique_imgs.clear()
        #ts_unique_imgs.clear()
        tr_image_names.clear()
        #ts_image_names.clear()
        print("CLEANED TR IMAGES ",tr_image_names)
        #print("CLEANED TS IMAGES ",ts_image_names)
        print("-----------Triggering the next task ------ TFRECORD generation --- SUCCESS")
        responce = requests.post("http://10.10.10.165:5000", json= {"message": "generate TF record"})


    # close the SFTP session
    sftp.close()


    return "Success"



if __name__ == '__main__':
    app.run(host='0.0.0.0')
