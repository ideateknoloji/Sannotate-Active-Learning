from flask import Flask, request
import paramiko
import json
import shutil
LABEL_STUDIO_URL = 'http://localhost:8080/'
LABEL_STUDIO_API_KEY = '1e68a65e957636b82e6bf92dad1e8c8a4cb14dd0'
from label_studio_sdk import Client
ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)


app = Flask(__name__)

@app.route("/receive", methods=["GET", "POST"])
def make_prediction():
    
    """
    Make predictions for unannotated images in Label Studio. 
    """
    response = request.get_data()
    response = json.loads(response)

    unannotated_task = response["task"]
    results = response["result"]
    score = response["score"]
    
    print("-------------------Label Studio Connection is OK---------------------------")
    project = ls.get_project(1)

    project.create_prediction(task_id=unannotated_task, model_version=None, result=str(results), score=score)
    print("PREDICTION IS SENT TO LABEL STUDIO FOR IMAGE ID {}".format(unannotated_task))

    return "SUCCESS"

@app.route("/copyfiles",methods=["POST"])
def copy_files():
    """
    COPY UNANNOTATED IMAGES TO SERVER and return unannotated images' IDs
    """
    
    print(ls.check_connection())
    project = ls.get_project(1)
    unlabeled_tasks_ids = project.get_unlabeled_tasks_ids()

    for i in range(len(unlabeled_tasks_ids)):
        image_path = "/home/jsultanov/.local/share/label-studio/media/upload/1/" + "".join(project.get_task(unlabeled_tasks_ids[i])["file_upload"])
        remote_path = "classification_un_images/" +"{}_".format(unlabeled_tasks_ids[i]) + "".join(project.get_task(unlabeled_tasks_ids[i])["file_upload"])
        shutil.copy(image_path, remote_path)
    print("-------------------------UNANNOTATED IMAGES COPIED FROM LOCAL TO SERVER----------------------")
    
    return "OK"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000")
