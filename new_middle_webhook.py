from flask import Flask, request
import pandas as pd
import os
import glob
import time
from pprint import pprint
import paramiko
app = Flask(__name__)
ssh = paramiko.SSHClient()
from label_studio_sdk import Client
from zipfile import ZipFile
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
import requests
# connect to the server
ssh.connect(hostname="10.10.10.165", username="jsultanov", password="Bil444islem..")
from ultralytics import YOLO
# create new pandas dataframe with annotations

if not os.path.exists("annotations.csv"):
    df = pd.DataFrame(columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
else:
    df = pd.read_csv("annotations.csv", error_bad_lines=False)

request_count = 0
limit =  2
annotations = []

LABEL_STUDIO_URL = 'http://localhost:8080'
#http://localhost:8080/projects/1/data?tab=1
API_KEY = '6d8ad45851a711b361dd3c4330f01b8358109b9d'

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()


"""
from label_studio.utils.io import BaseExporter

class YOLOExporter(BaseExporter):
    def dump(self, tasks, output_dir):
        for task in tasks:
            filename = task['filename']  # Assuming filename is stored in task
            annotations = task['annotations']  # Assuming annotations are stored in task
            
            with open(f"{output_dir}/{filename}.txt", "w") as f:
                for annotation in annotations:
                    label = annotation['label']
                    # Assuming label contains class name and bounding box coordinates
                    class_name = label['class_name']
                    xmin, ymin, xmax, ymax = label['xmin'], label['ymin'], label['xmax'], label['ymax']
                    
                    # Calculate YOLO format coordinates (normalized)
                    image_width, image_height = task['image_width'], task['image_height']
                    yolo_x = (xmin + xmax) / (2 * image_width)
                    yolo_y = (ymin + ymax) / (2 * image_height)
                    yolo_width = (xmax - xmin) / image_width
                    yolo_height = (ymax - ymin) / image_height
                    
                    # Write YOLO format line
                    f.write(f"{class_name} {yolo_x} {yolo_y} {yolo_width} {yolo_height}\n")

"""
"""
def upload_directory(local_path, remote_path, sftp):
    # Iterate through all files and subdirectories in local_path
    for root, dirs, files in os.walk(local_path):
        # Construct the remote path for the current directory
        remote_root = os.path.join(remote_path, os.path.relpath(root, local_path))

        #remote_root = remote_path + os.path.relpath(root,local_path)

        try:
            # Make the remote directory (recursively) if it doesn't exist
            sftp.mkdir(remote_root)
        except Exception as e:
            print(f"Failed to create remote directory {remote_root}: {e}")
        

        for file in files:
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.join(remote_root, file)
            try:
                # Upload each file to the remote directory
                sftp.put(local_file_path, remote_file_path)

                print(f"Uploaded {local_file_path} to {remote_file_path}")
            except Exception as e:
                print(f"Failed to upload {local_file_path}: {e}")
"""
def upload_directory(local_path, remote_path, sftp):
    # Replace Windows-style path separator (\) with Unix-style separator (/)
    remote_path = remote_path.replace('\\', '/')

    # Iterate through all files and subdirectories in local_path
    for root, dirs, files in os.walk(local_path):
        # Construct the remote path for the current directory
        relative_path = os.path.relpath(root, local_path).replace('\\', '/')
        remote_root = os.path.join(remote_path, relative_path)
        
        """
        try:
            # Make the remote directory (recursively) if it doesn't exist
            sftp.mkdir(remote_root)
        except Exception as e:
            print(f"Failed to create remote directory {remote_root}: {e}")
        """

        for file in files:
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.join(remote_root, file).replace('\\', '/')
            try:
                # Upload each file to the remote directory
                sftp.put(local_file_path, remote_file_path)
                print(f"Uploaded {local_file_path} to {remote_file_path}")
            except Exception as e:
                print(f"Failed to upload {local_file_path}: {e}")
                print("Server path is :" +remote_file_path)




@app.route("/",methods=["POST"])
def receive_webhook(df=df):
    """
    This function receives the webhook from Label Studio and saves the annotations to a csv file to be used for training the model 
    and also triggers the next task if the number of annotations reaches 10.
    """
    
    print("POST HAS BEEN ARRIVED!!!!!!!!")
    
    global request_count
    
    request_count +=1

    # open sftp connection
    sftp = ssh.open_sftp()


    # create an empty list to store the annotations
    objects = []


    # get the number of annotations
    num_annotations = len(request.get_json()["annotation"]["result"])

    if request_count >= limit:
        # Export annotations to YOLO format using Label Studio SDK
        PROJECT_ID = 1
        EXPORT_TYPE = 'YOLO'
        project = ls.get_project(PROJECT_ID)

        export_result = project.export_snapshot_create(
            title=f'export-test-{request_count}',
            task_filter_options={
                'view': 1,
                'finished': 'only',
                'annotated': 'only',
            }
        )
        export_id = export_result['id']

        # Wait until export snapshot is ready
        while project.export_snapshot_status(export_id).is_in_progress():
            time.sleep(1.0)

        """
        # Download the snapshot
        status, zip_file_path = project.export_snapshot_download(
            export_id=export_id,
            export_type=EXPORT_TYPE,
            path='.',
            #download_resources=True
        )

        """
        os.system("label-studio export 1 YOLO --export-path=C:/Users/ufuk.cefaker/Desktop/S-Annotate/Sannotate-Active-Learning-main/mydata")
        #os.system("label-studio export 1 -format=yolo --export-path=C:/Users/ufuk.cefaker/Desktop/S-Annotate/Sannotate-Active-Learning-main/mydata")

        #label-studio export <project_id> --format=yolo --output-dir=C:/Users/ufuk.cefaker/Desktop/S-Annotate/Sannotate-Active-Learning-main/mydata



        export_path = "C:/Users/ufuk.cefaker/Desktop/S-Annotate/Sannotate-Active-Learning-main/mydata"
        latest_zip_file = max(glob.glob(os.path.join(export_path, "*.zip")), key=os.path.getctime)

        print("Latest zip file:", os.path.basename(latest_zip_file))
        
        
        with ZipFile("mydata/"+os.path.basename(latest_zip_file) , 'r') as zip: 
            # extracting all the files 
            print('Extracting all the files now...') 
            zip.extractall(export_path) 
            print('Done!') 
        

        os.system("python splits.py")


        
        sftp.put('yolo_config.yaml',"yolo_config.yaml")


        """
        sftp.put('mydata/test',"train_media/test")

        sftp.put('mydata/train',"train_media/train")

        sftp.put('mydata/val',"train_media/val")
        
        
        #sftp.put('mydata/test/images',"test/")

        sftp.put('mydata/classes.txt',"train_media/classes.txt")

        sftp.put('mydata/notes.json',"train_media/notes.json")
        """

        try:
            # Upload the directory recursively
            upload_directory('mydata/test', "train_media/test", sftp)
            upload_directory('mydata/train', "train_media/train", sftp)
            upload_directory('mydata/val', "train_media/val", sftp)
            #buradan yolla!!!!!!!!!!!!!!!!!!

        finally:
            # Close the SFTP session and the transport
            sftp.close()


        #sftp.close()
        #sftp.put şeklinde yapıp extract ettiğin file ları server a yolla ve orda eğitime başlat.
        #ngrok çalıştırmayı unutma
        #label studio da

        responce = requests.post("http://10.10.10.165:5000", json= {"message": "generate YOLO training"})


        """
        print("YOLO STARTED TO TRAINING!!!!!!")
        # Load a COCO-pretrained YOLOv5n model
        model = YOLO("yolov5n.pt")

        # Display model information (optional)
        model.info()

        #Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="yolo_config.yaml", epochs=10, imgsz=640)
        """


        #print(f'Status of the export is {status}.\nFile path is {"A"}')

        # Additional logic after export (e.g., uploading files, triggering next tasks)...

        # Reset request count for the next batch
        request_count = 0

        return "Success"
    

    return "Success1"


if __name__ == '__main__':
    app.run(host='0.0.0.0')
