import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from PIL import Image
from collections import defaultdict
from io import StringIO
import json

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
cat_indexStr = open('cat_indexes', 'r')
category_index = json.loads(cat_indexStr.read())

from os import listdir
import numpy as np
import cv2
import json
with open('annotations/instances_val2014.json') as json_file:
    data = json.load(json_file)
    print([k for k in data])
valImageIds = {int(li.split('.')[0].split('_')[-1]): cv2.imread('val2014/'+li).shape for li in listdir('val2014') if int(li.split('.')[0].split('_')[-1]) in [da['image_id'] for da in data['annotations']]}
imgbboxes = {}
ground_truth = {}
 #if da['image_id'] in valImageIds
setvalImageIds = set(valImageIds)
for da in data['annotations']:
    if da['image_id'] in setvalImageIds:
        if not da['image_id'] in imgbboxes:
            imgbboxes[da['image_id']] = [da['bbox']]
        else:
            imgbboxes[da['image_id']].append(da['bbox'])
for da in data['annotations']:
    if da['image_id'] in setvalImageIds:
        if not da['image_id'] in ground_truth:
            ground_truth[da['image_id']] = { 'detection_boxes' : np.array([[da['bbox'][1],da['bbox'][0], da['bbox'][1] + da['bbox'][3], da['bbox'][0] + da['bbox'][2]]]), 'detection_classes' : np.array([da['category_id']])}
        else:
            ground_truth[da['image_id']]['detection_boxes'] = np.append(ground_truth[da['image_id']]['detection_boxes'], np.array([[da['bbox'][1], da['bbox'][0], da['bbox'][1] +  da['bbox'][3], da['bbox'][0] + da['bbox'][2]]]), axis = 0)
            ground_truth[da['image_id']]['detection_classes'] = np.append(ground_truth[da['image_id']]['detection_classes'], da['category_id'])
            

def intersects(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea > 0

import time

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
def calculate_IOU(model, image_path, bboxes):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    gt_bboxs = bboxes[int(image_path.split('/')[-1].split('.')[0].split('_')[-1])]
    detectedboxes = output_dict['detection_boxes']
    detectedboxes = [[dbx[0]*image_np.shape[0],dbx[1]*image_np.shape[1], dbx[2]*image_np.shape[0],dbx[3]*image_np.shape[1]] for dbx in detectedboxes]
    if len(detectedboxes) == 0 and len(gt_bboxs) == 0:
        return 1
    if len(detectedboxes) == 0:
        return 0
    return sum([sum([bb_intersection_over_union(gtb, dbb) for gtb in gt_bboxs if intersects(gtb, dbb)])/len([gtb for gtb in gt_bboxs if intersects(gtb, dbb)]) if len([gtb for gtb in gt_bboxs if intersects(gtb, dbb)]) > 0 else 0 for dbb in detectedboxes])/len(detectedboxes)
def calculate_optimized_IOU(infer_network, net_input_shape, image_path, bboxes):
    width, height, _ = infer_optimized(infer_network, net_input_shape, image_path)
    gt_bboxs = bboxes[int(image_path.split('/')[-1].split('.')[0].split('_')[-1])]
    if infer_network.wait() == 0:
        output = infer_network.extract_output()
        detectedboxes = [[int(o[3]*width), int(o[4]*height), int(o[5]*width), int(o[6]*height)] for o in output[0][0] if o[2] > 0.5]
        if len(detectedboxes) == 0 and len(gt_bboxs) == 0:
            return 1
        if len(detectedboxes) == 0:
            return 0
        return sum([sum([bb_intersection_over_union(gtb, dbb) for gtb in gt_bboxs if intersects(gtb, dbb)])/len([gtb for gtb in gt_bboxs if intersects(gtb, dbb)]) if len([gtb for gtb in gt_bboxs if intersects(gtb, dbb)]) > 0 else 0 for dbb in detectedboxes])/len(detectedboxes)
def check_pred(model, image_path, bboxes):
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    gt_bboxs = bboxes[int(image_path.split('/')[-1].split('.')[0])]
    detectedboxes = output_dict['detection_boxes']
    detectedboxes = [[dbx[0]*image_np.shape[0],dbx[1]*image_np.shape[1], dbx[2]*image_np.shape[0],dbx[3]*image_np.shape[1]] for dbx in detectedboxes]
    return detectedboxes, gt_bboxs
def inference_time(model, image_path):
    image_np = np.array(Image.open(image_path))
    startTime = time.time()
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    return time.time() - startTime
def infer_optimized(infer_network, net_input_shape, image_path):
    frame = np.array(Image.open(image_path))
    width = frame.shape[1]
    height = frame.shape[0]
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    startTime = time.time()
    infer_network.async_inference(p_frame)
    return (width, height, startTime)
def inference_optimized_time(infer_network, net_input_shape, image_path):
    _, _, startTime = infer_optimized(infer_network, net_input_shape, image_path)
    if infer_network.wait() == 0:
        return time.time() - startTime

from inference import Network
imgbboxes = { g:ground_truth[g]['detection_boxes'] for g in ground_truth}
infer_network = Network()
infer_network.load_model('frozen_inference_graph.xml', 'CPU', '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
net_input_shape = infer_network.get_input_shape()
ious = [calculate_optimized_IOU(infer_network, net_input_shape, image_path, imgbboxes) for image_path in ['val2014/'+li for li in listdir('val2014') if int(li.split('.')[0].split('_')[-1]) in ground_truth][:100]]
avgiou = sum(ious)/len(ious)
print('Average iou is '+str(avgiou))
infTimes = [inference_optimized_time(infer_network, net_input_shape, image_path) for image_path in ['val2014/'+li for li in listdir('val2014') if int(li.split('.')[0].split('_')[-1]) in ground_truth][:100]]
print('Average inference time is '+str(sum(infTimes)/len(infTimes)))