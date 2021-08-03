import sys
import cv2
import time
#import json     #***
import numpy as np
from argparse import ArgumentParser

camera_width  = 640                 #显示图像宽度
camera_height = 480                 #显示图像高度
start_time = 0                      #推断开始时间
end_time = 0                        #结束时间
total_time = 0                      #总时间
inf_image_cnt = 0                   #推断图片数
box_color = (0,0,255)               #识别框颜色BGR
box_thickness = 2                   #识别框粗细
label_text_color = (0,0,255)        #标签颜色BGR
label_text_size  = 0.8              #标签字号
inf_time = ""                       #推断速度显示

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

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-c", "--camera", help="Shooting equipment;USB_camera or laptop_camera(laptop by default)", default="laptop_camera", type=str)
    return parser

def main():
    global inf_time
    global inf_image_cnt
    global end_time
    global total_time
    #fffps = [] #***

    args = build_argparser().parse_args()
    status = "Without OpenVINO:CPU"

    # --------------------------------- 1. Opencv读取模型 -----------------------------------------------------
    pbfile = "pbmodels/frozen_inference_graph.pb"                           #tensorflow pb模型
    pbtxtfile = "pbmodels/frozen_inference_graph.pbtxt"                     #tensorflow pbtxt模型
    cvNet = cv2.dnn.readNetFromTensorflow(model=pbfile, config=pbtxtfile)   #opencv dnn加载模型
    cvNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)          
    cvNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # --------------------------------- 2. 初始化摄像头 -----------------------------------------------------
    if "laptop_camera" in args.camera :
        cap = cv2.VideoCapture(0)                       #0为笔记本内置摄像头，2为笔记本红外摄像头
    elif "USB_camera" in args.camera :
        cap = cv2.VideoCapture(4)                       #4为USB摄像头
    cap.set(cv2.CAP_PROP_FPS, 30)                       #初始化摄像头帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)     #初始化摄像头宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)   #初始化摄像头高度
    time.sleep(1)
    # -----------------------------------------------------------------------------------------------------

    # --------------------------------- 3. 读取摄像头数据并推断 ----------------------------------------------
    while cap.isOpened():     
        ret,image = cap.read()
        if not ret :
            break
        start_time = time.time()
        cvNet.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))      #opencv输入图像并开始推断
        cvOut = cvNet.forward()                         #输出推断结果
        output = cvOut[0][0]
        end_time = time.time() - start_time

        #去重叠框(可删除)
        for i in range(len(output)):
            if(output[i][2] < 0):
                output[i][2] = 0
            elif(output[i][2]==0):
                continue
            for j in range(i+1,len(output)):
                width_of_overlap_area = (min(output[i][5], output[j][5]) - max(output[i][3], output[j][3]))*camera_width
                height_of_overlap_area = (min(output[i][6], output[j][6]) - max(output[i][4], output[j][4]))*camera_height
                area_of_overlap = 0
                if (width_of_overlap_area <= 0 or height_of_overlap_area <= 0):
                    continue
                else:
                    area_of_overlap = width_of_overlap_area * height_of_overlap_area
                box1_area = (output[i][6]-output[i][4])*(output[i][5]-output[i][3])*camera_height*camera_width
                box2_area = (output[j][6]-output[j][4])*(output[j][5]-output[j][3])*camera_height*camera_width
                area_of_union = box1_area + box2_area - area_of_overlap
                if (area_of_union > 0 and (area_of_overlap / area_of_union)>=0.4):
                    output[j][2] = 0
                else :
                    continue

        #画图
        for obj in output: 
            if obj[2] > 0.5:
                label = np.int(obj[1])
                confidence = obj[2]
                xmin = np.int(camera_width  * obj[3])
                ymin = np.int(camera_height * obj[4])
                xmax = np.int(camera_width  * obj[5])
                ymax = np.int(camera_height * obj[6])
                label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),box_color,box_thickness)
                cv2.putText(image,label_text,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,label_text_size,label_text_color,2)
            else :
                continue
        cv2.putText(image, status,   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, inf_time, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Result", image)

        #记录结束时间，并计算帧率       
        total_time += end_time
        inf_image_cnt += 1
        if inf_image_cnt >=10 :
            inf_time = "Inference Speed:{:.3f} FPS".format(inf_image_cnt/total_time)
            total_time = 0
            inf_image_cnt = 0
            #fffps.append(inf_time)   #***

        #q键退出
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    # -----------------------------------------------------------------------------------------------------
    #with open("dete.json","w") as f:
    #    json.dump(fffps,f)          #***

    cv2.destroyAllWindows()

if __name__=="__main__":
    sys.exit(main() or 0)
