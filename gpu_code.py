import os
import shutil
from calculate_entropy import get_dataset_entropy, entropy_result_visualize
from PIL import Image
import json
import dlib
import gc
import tensorflow as tf
import requests
URL1 = "http://01f5-88-243-144-189.ngrok-free.app/copyfiles"
URL2 = "http://01f5-88-243-144-189.ngrok-free.app/receive"

def convert_to_ls(x, y, width, height, original_width, original_height):
    return x / original_width * 100.0, y / original_height * 100.0, \
           width / original_width * 100.0, height / original_height * 100

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

# Your existing GPU-related code, adapted to work as a standalone script

# transform annotations to TFRecord
os.system("python generate_tfrecord.py")
print("=====Generated TFRecord======")
os.system("python /home/jsultanov/models/research/object_detection/model_main_tf2.py  --model_dir='faster_rcnn_resnet_101/empty_model/' --pipeline_config_path='/home/jsultanov/exported-models/faster_rcnn_resnet_101V2/pipeline.config'")
print("========TRAINING FINISHED===========")
os.system("python /home/jsultanov/models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path '/home/jsultanov/faster_rcnn_resnet_101/pipeline.config' --trained_checkpoint_dir 'faster_rcnn_resnet_101/empty_model' --output_directory 'exported-models/faster_rcnn_resnet_101V2' ")
print("========MODEL IS EXPORTED AND READY FOR INFERENCE")

# delete all previous checkpoints. Because further training models needs empty folder to save all checkpoints
model_ckpnts = 'faster_rcnn_resnet_101/empty_model'
delete_files(model_ckpnts)

print("========DELETED ALL CHECKPOINTS========")
delete_files("train_media/")
delete_files("test_media/")

# make POST request to copy unannotated images from LOCAL to
# SERVER ------- DONE 
response = requests.post(url=URL1, json={"hi":"hello"})

model_path = "/home/jsultanov/exported-models/faster_rcnn_resnet_101V2/saved_model"
folder = "unannotated_imgs/"
image_paths = []

# get the list of images' paths
for filename in os.listdir(folder):
    if filename.endswith(".JPG") or filename.endswith(".jpg"):
        image_paths.append(os.path.join(folder, filename))

# Calculate entropy result for each image in unannotated image in folder
entropy_result = get_dataset_entropy(IMAGE_PATH=folder, MODEL_PATH=model_path)
max_10_entropies = entropy_result[:10, :]

result_dict = {}
for row in max_10_entropies:

    image_name = image_paths[int(row[0])]
    im_w, im_h = Image.open(image_name).size
    index_of_slash = image_name.index('/')
    task_id = image_name[index_of_slash + 1:].split("_")[0]
    results = entropy_result_visualize(image_path=image_name, model=model_path)
    results = [sublist for sublist in results if sublist.all()]

    payload_list = []

    for result in results:
        label, xmin, ymin, xmax, ymax, score = result
        rect = dlib.rectangle(xmin, ymin, xmax, ymax)
        x = (rect.left() / im_w) * 100
        y = (rect.top() / im_h) * 100
        width = (rect.right() - rect.left()) / im_w * 100
        height = (rect.bottom() - rect.top()) / im_h * 100
        if label == 1:
            class_name = "Sise"
        else:
            class_name = "Kutu"

        payload_dict = {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "rectanglelabels": [class_name],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        }
        payload_list.append(payload_dict)

    result_dict["annotation"] = {
        "task_id": task_id,
        "objects": payload_list}

    response = requests.post(url=URL2, json=result_dict)
    print("========PREDICTION IS SENT FOR {} ==========".format(task_id))

delete_files(folder)
print("========CLEANED UNANNOTATED IMAGE FOLDER========")

# MAKE INFERENCE TO CALCULATE mAP metric on test set
folder = "test_imgs/"
image_paths = []
# get the list of images' paths
for filename in os.listdir(folder):
    if filename.endswith(".JPG") or filename.endswith(".jpg"):
        image_paths.append(os.path.join(folder, filename))

for image_name in image_paths:
    im_w, im_h = Image.open(image_name).size
    results = entropy_result_visualize(image_path=image_name, model=model_path)
    results = [sublist for sublist in results if sublist.all()]
    real_img_name = image_name.split("/")[-1]

    with open(f"Object-Detection-Metrics/detections/{real_img_name}.txt", 'w') as file:
        for result in results:
            label, xmin, ymin, xmax, ymax, score = result
            if label == 1:
                class_name = "Sise"
            else:
                class_name = "Kutu"
            line = f"{class_name} {score} {xmin} {ymin} {xmax} {ymax}\n"
            file.write(line)


os.system("python Object-Detection-Metrics/pascalvoc.py")

# Clear the TensorFlow session and collect garbage to free GPU memory
tf.keras.backend.clear_session()
gc.collect()
