import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np 
import pylab
import cv2 
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      # specify type here - bbox 类型

useopenvino = ""              #是否使用openvino "" "no"
device_precision = "_NCS"     #"" "_NCS" "_CPU_INT8" "_CPU_FP16"          

annFile='/home/jmj/project/COCO/annotations/instances_val2017.json'
cocoGt=COCO(annFile)


resFile = '/home/jmj/project/SSD-MobileNetv2-OpenVINO/{}openvino_result{}.json'.format(useopenvino,device_precision)
cocoDt = cocoGt.loadRes(resFile)


imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:200]
#imgIds = imgIds[np.random.randint(100)]

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()