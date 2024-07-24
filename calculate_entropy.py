# %%
import os
from shutil import copy

#import cv2


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# %%
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)]
    )

# %%
IMAGE_PATH = "unannotated_imgs/"
MODEL_PATH = "/home/jsultanov/exported-models/faster_rcnn_resnet_101V1/saved_model"

# %%
def pred_boxes(img: Image, boxes: np.ndarray, scores: np.ndarray,
                              class_name,
                              score_threshold=0.1):
    #image_name = image_name.split(".")[0]
    img_size = img.size
    im_width, im_height = img_size

    row, col = boxes.shape

    box_coordinates = []
    count = 0
    for r in range(row):
        if scores[r] >= score_threshold:
            ymin = boxes[r, 0]
            xmin = boxes[r, 1]
            ymax = boxes[r, 2]
            xmax = boxes[r, 3]

            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
            print("BBOXES  ",left,right,top,bottom)
            img = np.array(img)
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            # cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 1)
            
            
            count += 1
            box_coordinates.append((class_name[r], left, top, right, bottom))

    
    return np.array(box_coordinates)

    # return np.array(box_coordinates)


# %%
def detection_result(image: Image, model) -> np.array:
    image = np.array(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    detection_scores = detections["detection_scores"]
    return detection_scores


# %%
def entropy(score: np.array) -> float:
    boxes_score = score[score > 0.50]  # Simplified the slicing condition
    return -np.nansum(np.multiply(boxes_score, np.log(boxes_score)))

"""
def entropy(score: np.array) -> np.float64:
    boxes_score = score[score > 0.50]  # Simplified the slicing condition
    return -np.nansum(np.multiply(boxes_score, np.log(boxes_score)))

def entropy(score: np.array) -> np.float():
    boxes_score = score[score[:] > 0.50]
    return -np.nansum(np.multiply(boxes_score, np.log(boxes_score)))
"""

# %%
def calculate_entropy_result(image_list, model):
    entropy_list = []
    origin_index = np.arange(0, len(image_list))

    for i, image_name in enumerate(image_list):
        
        # image_name = image_list[0]
        img = Image.open(IMAGE_PATH + image_name)

        detection_scores = detection_result(image=img, model=model)
        ent = entropy(score=detection_scores)
        #print(i, image_name, ent)
        entropy_list.append(ent)

    uncertainty = np.column_stack((origin_index, entropy_list))

    return uncertainty[(-uncertainty[:, 1]).argsort()]


# %%

def entropy_result_visualize(image_path: str, model):
    # image_no = 18
    img1 = Image.open(image_path)
    DETECTION = tf.saved_model.load(export_dir=model)
    img = np.array(img1)
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = DETECTION(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    coordinates = pred_boxes(img=img1,
                              boxes=detections["detection_boxes"],
                              scores=detections["detection_scores"],
                              class_name=detections['detection_classes']
                              )
    return coordinates

# %%

def sent_max_entropy_image_to_oracle(entropy_result, image_list):
    send_oracle_image_number = entropy_result[:10, :]
    print(send_oracle_image_number)

    '''
    for img_number in send_oracle_image_number:
        image_name = image_list[int(img_number)]
        gt_name = image_name.replace("JPG", "txt").replace("jpg", "txt")

        print("{image_name} sent to ORACLE ".format(image_name=image_name))

        # copy image file
        copy(src=IMAGE_PATH + image_name, dst=LABELED_DATA_PATH + image_name)
        os.remove(path=IMAGE_PATH + image_name)

        # copy gt file
        copy(src=IMAGE_PATH + gt_name, dst=LABELED_DATA_PATH + gt_name)
        os.remove(path=IMAGE_PATH + gt_name)

'''
# %%

def get_dataset_entropy(IMAGE_PATH = IMAGE_PATH, MODEL_PATH=MODEL_PATH):
    #category_index = {1: {'id': 1, 'name': 'Sise'}, 2: {'id': 2, 'name': 'Kutu'}}

    image_list = [img for img in os.listdir(IMAGE_PATH) if img.endswith((".jpg", "JPG"))]

    DETECTION_MODEL = tf.saved_model.load(export_dir=MODEL_PATH)

    entropy_score_list = calculate_entropy_result(image_list=image_list, model=DETECTION_MODEL)

    # entropy_result_visualize(image_no=0, entropy_score=entropy_score_list, model=DETECTION_MODEL, image_list=image_list)

    sent_max_entropy_image_to_oracle(entropy_result=entropy_score_list, image_list=image_list)
    print(entropy_score_list.shape)
    return entropy_score_list

if __name__ == '__main__':
    get_dataset_entropy(IMAGE_PATH, MODEL_PATH)
