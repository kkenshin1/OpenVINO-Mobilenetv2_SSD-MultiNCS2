# OpenVINO-Mobilenetv2_SSD-MultiNCS2
OpenVINO 2019_R3.1 + MobileNetV2-SSD + Intel CPU/multi-NCS2 +USB_camera<br><br>

## Overview
Based on OpenVINO, this project implement the model optimization and inference acceleration of MobileNetV2-SSD. This project uses the model optimizer and calibration tool to achieve the FP16/FP32
compression and INT8 quantization of the algorithm, and evaluates the overall accuracy of the
model on the COCO dataset. Moreover, we propose a multi-accelerator parallel acceleration scheme to detect video stream data in real time by runing inference engine code.

This project is inspired by [PINTO0309/MobileNet-SSD-RealSense](https://github.com/PINTO0309/MobileNet-SSD-RealSense) and optimize for his multi-stick code.<br><br>

## Download Model
You can download Tensorflow MobileNetV2-SSD Model [here](download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) and place the pbmodels in the &lt;pbmodels/&gt; floder.

[OpenVINO Model Optimizer Website](https://docs.openvinotoolkit.org/2019_R3.1/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) also provides lots of pre-trained models.<br><br>

## Model Optimization
### FP16/FP32
Run mo.py script to generate the FP16/FP32 IR file
```bash
$ cd OpenVINO-Mobilenetv2_SSD-MultiNCS2
$ sudo python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ./pbmodels/frozen_inference_graph.pb \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json\
--tensorflow_object_detection_api_pipeline_config ./pbmodels/pipeline.config \
--output_dir ./lrmodels/FP32 \
--data_type FP32
```
or
```bash
$ cd OpenVINO-Mobilenetv2_SSD-MultiNCS2
$ sudo python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ./pbmodels/frozen_inference_graph.pb \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json\
--tensorflow_object_detection_api_pipeline_config ./pbmodels/pipeline.config \
--output_dir ./lrmodels/FP16 \
--data_type FP16
```
### INT8
OpenVINO can run [low-precision inference](https://docs.openvinotoolkit.org/2019_R3.1/_docs_IE_DG_Int8Inference.html) using Intel MKL-DNN. 

We can run calibrate.py script in simplified mode to generate the INT8 IR file.

This project uses FP16 IR file as input of this script.

In OpenVINO 2019_R3.1, only cpu can run in low precision inference.

```bash
$ cd OpenVINO-Mobilenetv2_SSD-MultiNCS2
$ sudo python3 /opt/intel/openvino/deployment_tools/tools/calibration_tool/calibrate.py \
-sm \
-m ./lrmodels/FP16/frozen_inference_graph.xml \
-s <path-to-your-dataset> \
-e ./libcpu_extension.so \
-td CPU \
--output_dir ./lrmodels/INT8
```
Here is the Low-Precision 8-bit Integer Inference Workflow

![image](https://github.com/kkenshin1/OpenVINO-Mobilenetv2_SSD-MultiNCS2/blob/main/imgs/cpu_int8_flow.png)<br><br>

## Single CPU/NCS2 Inference
You can run `opencv_dnn_ssd.py` or `openvino_singlestick_sync.py` script to implement the inference of MobileNetV2-SSD on single CPU/NCS2. You can also choose different precision format of model.

You should generate .pbtxt file before run `opencv_dnn_ssd.py` script.

```bash
$ cd OpenVINO-Mobilenetv2_SSD-MultiNCS2
$ python3 opencv_dnn_ssd.py -c USB_camera
```

You should generate IR file before run `openvino_singlestick_sync.py` script.

```bash
$ cd OpenVINO-Mobilenetv2_SSD-MultiNCS2
$ python3 openvino_singlestick_sync.py -h
usage: openvino_singlestick_sync.py [-h] [-d DEVICE] [-c CAMERA]
                                    [-p PRECISION]
optional arguments:
  -h, --help    show this help message and exit
  -d, --device
                Specify the target device to infer on; CPU or MYRIAD
                is acceptable. Sample will look for a suitable plugin
                for device specified (CPU by default)
  -c, --camera
                Shooting equipment;USB_camera or laptop_camera(laptop
                by default)
  -p, --precision
                Model Precision; FP32 or FP16 or INT8 (FP16 by
                default)

$ python3 openvino_singlestick_sync.py -d CPU -c USB_camera -p FP16
```
<br><br>



## Precision Evaluation


## Multi-NCS2 Parallel Inference


## Multi-NCS2 Parallel Inference Optimization


## Experimental Results
### Precision and Single-stick Speed

### Multi-NCS2 Inference Speed


## Reference