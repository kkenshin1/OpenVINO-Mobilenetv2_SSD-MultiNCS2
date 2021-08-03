import sys
import os
import cv2
import time
import heapq
import numpy as np
import multiprocessing as mp
from openvino.inference_engine import IENetwork, IECore

camera_width = 1280                 #摄像头宽度
camera_height = 720                 #摄像头高度
frameBuffer = None                  #图像队列
results = None                      #结果队列
process = []                        #进程列表
req_list = []                       #请求列表
box_color = (0,0,255)               #识别框颜色BGR
box_thickness = 2                   #识别框粗细
label_text_color = (0,0,255)        #标签颜色BGR
label_text_size  = 1.0              #标签字号
res = None                          #当前推断结果
lastres = None                      #上一次推断结果
imagecnt = 0                        #总图片计数
detecnt = 0                         #推断图片计数
start_time = 0                      #开始时间
end_time = 0                        #结束时间
reappear_time = 0                   #显示总时间
detec_time = 0                      #推断总时间
reappear_fps = ""                   #显示帧率
detec_fps = ""                      #推断帧率
ncs_num = ""                        #计算棒数量
request_num = ""

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

# --------------------------- 获取所有计算棒编号，并转为多设备形式 ------------------------------
def achieve_device(ie):
    alldevice = "MULTI:"
    ncs_id=ie.get_metric("MYRIAD", "AVAILABLE_DEVICES")
    for i in range(0,len(ncs_id)):
        alldevice += "MYRIAD." + ncs_id[i]
        if i < len(ncs_id)-1 :
            alldevice += ","
    return alldevice, len(ncs_id) * 4           #*3表示每根计算棒3个推断请求
# ------------------------------------------------------------------------------------------

# -------------- 寻找请求列表索引 --------------------
def searchlist(l,x,notfound=-1):
    if x in l:
        return l.index(x)
    else:
        return notfound
# -------------------------------------------------

# -------------------------------------- 画出推断结果 ----------------------------------------
def Detection_Output(image,output):

    try :

        if isinstance(output, type(None)):
            return image

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
        
        cv2.putText(image, reappear_fps, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image,    detec_fps, (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image,      ncs_num, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image,  request_num, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

        return image

    except:
        import traceback
        traceback.print_exc()
# ------------------------------------------------------------------------------------------

# --------------------------------- 多进程1：摄像头 ------------------------------------------
def camera(frameBuffer,results):
    global res 
    global lastres 
    global detecnt 
    global imagecnt 
    global start_time 
    global end_time 
    global reappear_time
    global detec_time
    global detec_fps 
    global reappear_fps
    global ncs_num
    global request_num

    ie2 = IECore()
    alldevice2,num_req2 = achieve_device(ie2)
    ncs_num = "NCS_number: {}".format(num_req2/3)   #计算棒数量，根据推断请求修改数字
    request_num = "eachNCS request number: 3"       #修改每根计算棒推断请求时更改该数字

    cap = cv2.VideoCapture(4)                           #0为笔记本内置摄像头，2为笔记本红外摄像头，4为USB摄像头
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)     #初始化摄像头宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)   #初始化摄像头高度
    cap.set(cv2.CAP_PROP_FPS, 60)                       #初始化摄像头帧率

    while True :
        start_time = time.perf_counter()
        ret, color_image = cap.read()
        if not ret :
            continue
        if frameBuffer.full():
            frameBuffer.get()
        images = color_image
        frameBuffer.put(color_image.copy())
        res = None

        if not results.empty():
            res = results.get(False)
            detecnt += 1
            image = Detection_Output(images,res)
            lastres = res
        else :
            image = Detection_Output(images,lastres)
        
        cv2.imshow("Result", image)

        imagecnt += 1
        if imagecnt >= 50:
            reappear_fps = "Reappear: {:.3f}FPS".format(reappear_time/imagecnt)
            detec_fps    = "Inference: {:.3f}FPS".format(detecnt/detec_time)
            imagecnt = 0
            detecnt = 0
            reappear_time = 0
            detec_time = 0
        end_time = time.perf_counter() - start_time
        reappear_time += 1.0/end_time
        detec_time += end_time 

        #q键退出
        if cv2.waitKey(1)&0xFF == ord('q'):
            sys.exit(0)
# ------------------------------------------------------------------------------------------

# ----------------------------------- 多进程2：推断 ------------------------------------------
def inference(frameBuffer,results): 

    heap_request = []           #heapq堆
    infer_cnt = 0               #推断次数

    model_xml = "lrmodels/FP16/frozen_inference_graph.xml"  #FP16格式，适用于CPU与NCS2
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)

    ie = IECore()  
    input_blob = next(iter(net.inputs))
    input_h = net.inputs[input_blob].shape[2]      
    input_w = net.inputs[input_blob].shape[3]      
    out_blob = next(iter(net.outputs))
    alldevice,num_req = achieve_device(ie)
    request_list = [0]*num_req       #推断请求列表
    exec_net = ie.load_network(network=net, device_name=alldevice, num_requests=num_req)

    while True : 
        
        try:
            
            if frameBuffer.empty():
                continue
            image = frameBuffer.get()
            resized_image = cv2.resize(image, (input_w,input_h), interpolation=cv2.INTER_CUBIC)     #转换输入尺寸
            resized_image = resized_image[np.newaxis, :, :, :]              #增加维度
            images = resized_image.transpose((0, 3, 1, 2))                  #NHWC转NCHW

            req_id = searchlist(request_list, 0)
            if req_id > -1 :
                exec_net.start_async(request_id=req_id, inputs={input_blob: images})   #异步推断
                request_list[req_id] = 1
                infer_cnt += 1 
                if infer_cnt == sys.maxsize :
                    request_list = [0]*num_req
                    heap_request = []
                    infer_cnt = 0
                heapq.heappush(heap_request,(infer_cnt,req_id))
            
            cnt,dev = heapq.heappop(heap_request)
            if exec_net.requests[dev].wait(0) == 0 :
                exec_net.requests[dev].wait(-1)
                outputs = exec_net.requests[dev].outputs[out_blob]
                output = outputs[0][0]
                results.put(output)
                request_list[dev] = 0
            else :
                heapq.heappush(heap_request,(cnt,dev))
       
        except:
            import traceback
            traceback.print_exc()
# ---------------------------------------------------------------------------------------------------

# --------------------------------- 主进程：创建多进程与进程队列 -----------------------------------------
if __name__== "__main__":

    try :
        
        mp.set_start_method('forkserver')
        #多进程队列
        frameBuffer = mp.Queue(10)      #图像队列
        results = mp.Queue()          #推断结果队列

        #初始化多进程1：camera
        p1 = mp.Process(target=camera, args=[frameBuffer,results], daemon=True)

        #初始化多进程2：inference
        p2 = mp.Process(target=inference, args=[frameBuffer,results], daemon=True)

        p1.start()
        process.append(p1)
        p2.start()
        process.append(p2)

        while True:
            time.sleep(1)

    except:
        import traceback
        traceback.print_exc()
    
    finally:
        for p in range(len(process)):
            process[p].terminate()
        print ("FINISHED!")
# ----------------------------------------------------------------------------------------------------
