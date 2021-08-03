import sys
import os
import cv2
import time
import json
import pylab 
import numpy as np
import matplotlib.pyplot as plt 
from pycocotools.coco import COCO
from openvino.inference_engine import IENetwork, IECore
pylab.rcParams['figure.figsize']=(8.0,10.0)

# -------------------------------------- COCO_labels -------------------------------------------------
LABELS={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',            
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
        77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
        82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
        88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
# -----------------------------------------------------------------------------------------------------

box_color = (121,101,20)            #识别框颜色BGR
box_thickness = 2                   #识别框粗细
label_text_color = (121,101,20)     #标签颜色BGR
label_text_size  = 0.8              #标签字号
res = {}                            #当前结果
total_res=[]                        #总结果

#加载标注
dataDir='/home/jmj/project/COCO'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

#加载图像
imgIds = sorted(coco.getImgIds())  
imgIds = imgIds[0:200]
# loadImgs() 返回的是只有一个元素的列表, 使用[0]来访问这个元素

pbfile = "pbmodels/frozen_inference_graph.pb"                           #tensorflow pb模型
pbtxtfile = "pbmodels/frozen_inference_graph.pbtxt"                     #tensorflow pbtxt模型
cvNet = cv2.dnn.readNetFromTensorflow(model=pbfile, config=pbtxtfile)   #opencv dnn加载模型
cvNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)          
cvNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

for i in range(200):
    # 列表中的这个元素又是字典类型, 关键字有: ["license", "file_name", 
    #  "coco_url", "height", "width", "date_captured", "id"]
    img = coco.loadImgs(imgIds[i])[0]
    # 1) 使用本地路径, 对应关键字 "file_name"
    image = cv2.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
    width = image.shape[1]
    height = image.shape[0]
    resized_image = cv2.resize(image, (300,300), interpolation=cv2.INTER_CUBIC)     #转换输入尺寸
    cvNet.setInput(cv2.dnn.blobFromImage(resized_image, size=(300, 300), swapRB=True, crop=False))      #opencv输入图像并开始推断
    cvOut = cvNet.forward()                         #输出推断结果
    output = cvOut[0][0]

    #画图
    for obj in output: 
        if obj[2] > 0:
            label = np.int(obj[1])
            confidence = obj[2]
            xmin = np.int(width  * obj[3])
            ymin = np.int(height * obj[4])
            xmax = np.int(width  * obj[5])
            ymax = np.int(height * obj[6])
            x = round(width*obj[3],2)
            y = round(height*obj[4],2)
            res_width = round(width*(obj[5]-obj[3]),2)
            res_height = round(height*(obj[6]-obj[4]),2)
            score = float('%.2f'%confidence)   
            label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),box_color,box_thickness)
            cv2.putText(image,label_text,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,label_text_size,label_text_color,1)
            res = {"image_id":img['id'],"category_id":label,"bbox":[x,y,res_width,res_height],"score":score}
            total_res.append(res)
        else :
            continue

    cv2.imshow("Result", image)

with open('noopenvino_result.json','w') as f:
    json.dump(total_res,f)