import gc
import os
import random
import warnings
import numpy
import statistics
from shutil import copy
from subprocess import Popen
import run
import requests
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from flask import Flask, request
warnings.filterwarnings('ignore')
app = Flask(__name__)
import json
request_count = 0
num_epocs_to_train_in_each_step = 100
training_steps = 0.027
random_seed = 0
experiment_pretrain_model = "bert-base-cased"

sh_executable = "/bin/sh"
data_dir = "/home/nuhhatipoglu/SAnnotate-NLP-v2/data/DataSet-1"
initial_dir = "/home/nuhhatipoglu/SAnnotate-NLP-v2/NER_initial"
trials_dir = "/home/nuhhatipoglu/SAnnotate-NLP-v2/run_base_dir/6500"
beginning_identifier = "B-"  # Label beginning identifier
no_label_label = "O"
device = 0
URL = "http://10.10.10.165:8000"
def read_sentences_from_files(filepaths):
    sentences = []
    task_ids = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8-sig') as file:
            sentences.append(file.read().strip())
        task_id = int(filepath.split("/")[-1].split(".")[0])  # Extract task ID from the filename
        task_ids.append(task_id)
    return sentences, task_ids

def predict_ner_for_sentence(model, tokenizer, sentence):
    inputs = tokenizer.encode_plus(sentence, return_offsets_mapping=True, return_tensors="pt")
    offsets_mapping = inputs.pop("offset_mapping")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    token_predictions = numpy.argmax(logits, axis=-1)[0]
    scores = numpy.exp(logits) / numpy.exp(logits).sum(axis=-1, keepdims=True)
    
    labels = [model.config.id2label[pred] for pred in token_predictions]
    
    # Remove the special tokens (CLS and SEP)
    offsets_mapping = offsets_mapping.squeeze(0).tolist()[1:-1]
    token_predictions = token_predictions[1:-1]
    labels = labels[1:-1]

    # Combine offsets with labels
    labeled_offsets = [(label, start, end) for (start, end), label in zip(offsets_mapping, labels) if label != 'O']

    return token_predictions, scores, labeled_offsets

def entropy_score(results):
    predictions = results[0]
    scores = results[1]

    score = -numpy.nansum(numpy.multiply(scores, numpy.log(scores)))

    return score

@app.route("/", methods=["POST", "GET"])
def predict():
    global request_count
    #request.get_json(force=True, silent=True)
    
    # get dataset from initialize_dataset_v2() function
    test_data, train_data, dev_data = run.load_test_dev_train()
    current_train = run.convert_sentence_based_to_txt_based(train_data)
    test = run.convert_sentence_based_to_txt_based(test_data)
    current_dev = run.convert_sentence_based_to_txt_based(dev_data)
    
    print("=====Dataset loaded=======") # DONE
    # train model with received annotated data from Label Studio
    # log INFO to console about training process
    environment_name = "NER_" + str(request_count)

    # Create the environment and put files in it
    run.create_new_environment(environment_name)

    # Create train/dev/test files in the environment
    run.create_train_eval_test_files_in_environment(environment_name,
                                                {
                                                    "test": test,
                                                    "train": current_train,
                                                    "dev": current_dev
                                                })

    # Start training in the environment
    run.train_in_environment(environment_name=environment_name,
                            num_epochs=num_epocs_to_train_in_each_step,
                            model_name=experiment_pretrain_model,
                            random_seed=random_seed)
    run.get_results_from_environment(environment_name)
    print("=====Model trained=======")
    # Load the trained model for active learning sample selection
    #model = run.load_model(environment_name)

    
    request_count += 1
    # make POST request to Label Studio to get unannotated images
    requests.post("http://10.10.10.165:8000/copyfiles", json={"data":"1"})
    # convert unannotated images to numpy array using unannotated_dataset() function
    
 
    print("====Unannotated images loaded====")
    # make predictions on the unlabeled data from "unannotated images" folder, find their entropy and rank them and select the top n_samples and show corresponding image's index
    model = AutoModelForTokenClassification.from_pretrained(trials_dir + "/" + environment_name + "/model/")
    tokenizer = AutoTokenizer.from_pretrained(trials_dir + "/" + environment_name + "/model/")

    filepaths = [os.path.join("ner", file) for file in os.listdir("ner")]
    sentences,task_ids = read_sentences_from_files(filepaths)
    results_list = []


    for sentence,task_id in zip(sentences, task_ids):
        results = predict_ner_for_sentence(model, tokenizer, sentence)
        entropy = entropy_score(results)
        labeled_offsets = results[2]
        results_list.append((task_id, sentence, entropy, labeled_offsets))


    results_list.sort(key = lambda x: x[2], reverse = True)
    top_10_sentences = results_list[:10]
    result_dict = dict()
    for task_id, sentence, entropy, labeled_offsets in top_10_sentences:
        if len(labeled_offsets) != 0:
            payload_list = []
            for labeled_offset in labeled_offsets:
                print(f"Sentence: {sentence}")
                print(f"Labels with start and end positions: {labeled_offsets}")
                print(f"Entropy: {entropy}")
                label, start, end = labeled_offset
                payload_dict = {
                    
                        'from_name': 'label',
                        'to_name': 'text',
                        'type': 'labels',
                        'value': {
                            'start': start,
                            'end': end,
                            'text': sentence[start:end],
                            'labels': [label]
                        }
            }
                payload_list.append(payload_dict)
            result_dict["annotation"] = {
                    "score": 1,
                    "task_id" : task_id,
                    "objects": payload_list}
            data = json.dumps(payload_dict)
            headers = {"Content-Type": "application/json"}
            requests.post("http://10.10.10.165:8000/receive", json=result_dict, headers=headers)
            payload_dict.clear()
            print("=====Predictions made for {}=====".format(task_id))
    
 
    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8090)