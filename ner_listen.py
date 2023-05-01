from flask import Flask, request
import json
import shutil
LABEL_STUDIO_URL = 'http://10.10.10.165:8080/'
LABEL_STUDIO_API_KEY = 'e6f228316436728731ea29af029c3c9aee4ebb25'
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
    # print(json.dumps(response, indent = 1))

    unannotated_task = response["task"]
    results = response["result"]
    score = response["score"]
    # print(json.dumps(results, indent = 1))
    print("-------------------Label Studio Connection is OK---------------------------")
    project = ls.get_project(23)

    project.create_prediction(task_id=unannotated_task, model_version=None, result=[results], score=score)
    print("PREDICTION IS SENT TO LABEL STUDIO FOR TEXT ID {}".format(unannotated_task))


    return "SUCCESS"

@app.route("/copyfiles", methods=["POST"])
def copy_files():
    """
    COPY UNANNOTATED IMAGES TO SERVER and return unannotated images' IDs
    """
    print(ls.check_connection())
    project = ls.get_project(23)
    unlabeled_tasks_ids = project.get_unlabeled_tasks_ids()

    for i in range(len(unlabeled_tasks_ids)):
        # from each unlabeled task's text create txt file with task id in its name and copy it to server
        task = project.get_task(unlabeled_tasks_ids[i])
        text = task['data']['text']
        task_id = task['id']
        with open(f'NER/{task_id}.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Created {task_id}.txt')
        remote_path = "ner/" + f"{task_id}.txt"
        shutil.copy(f'NER/{task_id}.txt', remote_path)
    print("-------------------------UNANNOTATED IMAGES COPIED FROM LOCAL TO SERVER----------------------")
    
    return "OK"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000")
