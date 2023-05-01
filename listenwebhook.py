from flask import Flask, request
import paramiko
import json
LABEL_STUDIO_URL = 'http://localhost:8080/'
LABEL_STUDIO_API_KEY = 'e6f228316436728731ea29af029c3c9aee4ebb25'
from label_studio_sdk import Client
ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# connect to the server
ssh.connect(hostname="10.10.10.165", username="jsultanov", password="Bil444islem..")

app = Flask(__name__)

@app.route("/receive", methods=["GET", "POST"])
def make_prediction():
    
    """
    Make predictions for unannotated images in Label Studio. 
    """
    response = request.get_data()
    response = json.loads(response)
    #print(json.dumps(response, indent = 1))

    unannotated_task = response["annotation"]["task_id"]
    results = response["annotation"]["objects"]
    # print(json.dumps(results, indent = 1))
    
    print("-------------------Label Studio Connection is OK---------------------------")
    project = ls.get_project(18)

    project.create_prediction(task_id=unannotated_task, model_version=None, result=results)
    print("PREDICTION IS SENT TO LABEL STUDIO FOR IMAGE ID {}".format(unannotated_task))


    return "SUCCESS"

@app.route("/copyfiles",methods=["POST"])
def copy_files():
    """
    COPY UNANNOTATED IMAGES TO SERVER and return unannotated images' IDs
    """
    hi = request.get_json("hi")
    sftp = ssh.open_sftp()
    print(ls.check_connection())
    project = ls.get_project(18)
    unlabeled_tasks_ids = project.get_unlabeled_tasks_ids()

    for i in range(len(unlabeled_tasks_ids)):
        image_path = "C:\\Users\\jelal.sultanov\\AppData\\Local\\label-studio\\label-studio\\media\\upload\\18\\" + "".join(project.get_task(unlabeled_tasks_ids[i])["file_upload"])
        remote_path = "unannotated_imgs/" +"{}_".format(unlabeled_tasks_ids[i]) + "".join(project.get_task(unlabeled_tasks_ids[i])["file_upload"])
        sftp.put(image_path, remote_path)
    print("-------------------------UNANNOTATED IMAGES COPIED FROM LOCAL TO SERVER----------------------")
    
    return "OK"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000")
