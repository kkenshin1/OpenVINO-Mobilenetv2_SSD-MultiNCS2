# OpenVINO-Mobilenetv2_ssd-multiNCS2
OpenVINO 2019_R3.1 + MobileNetV2-SSD + Intel CPU/multi-NCS2 +USB_camera 
<br>
## Overview
Based on OpenVINO, this project implement the model optimization and inference acceleration of MobileNetV2-SSD. This project uses the model optimizer and calibration tool to achieve the FP16/FP32
compression and INT8 quantization of the algorithm, and evaluates the overall accuracy of the
model on the COCO dataset. Moreover, we propose a multi-accelerator parallel acceleration scheme to detect video stream data in real time by runing inference engine code.

This project is inspired by [PINTO0309/MobileNet-SSD-RealSense](https://github.com/PINTO0309/MobileNet-SSD-RealSense) and optimize for his multi-stick code.

<br>
## Download Model
You can download Tensorflow MobileNetV2-SSD Model [here](download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) and place the pbmodels in the &lt;pbmodels/&gt; floder.

[OpenVINO Model Optimizer Website](https://docs.openvinotoolkit.org/2019_R3.1/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) also provides lots of pre-trained models.

<br>
## Model Optimization
### FP16/FP32
Run mo.py script to generate the FP16/FP32 IR file
```bash
$ cd OpenVINO-Mobilenetv2_ssd-multiNCS2
$ sudo python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ./pbmodels/frozen_inference_graph.pb \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json\
--tensorflow_object_detection_api_pipeline_config ./pbmodels/pipeline.config \
--output_dir ./lrmodels/FP32 \
--data_type FP32
```
or
```bash
$ cd OpenVINO-Mobilenetv2_ssd-multiNCS2
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
$ cd OpenVINO-Mobilenetv2_ssd-multiNCS2
$ sudo python3 /opt/intel/openvino/deployment_tools/tools/calibration_tool/calibrate.py \
-sm \
-m ./lrmodels/FP16/frozen_inference_graph.xml \
-s <path-to-your-dataset> \
-e ./libcpu_extension.so \
-td CPU \
--output_dir ./lrmodels/INT8
```
Here is the Low-Precision 8-bit Integer Inference Workflow

![image](https://github.com/kkenshin1/OpenVINO-Mobilenetv2_ssd-multiNCS2/blob/main/imgs/cpu_int8_flow.png)

## Single CPU/NCS2 Inference



## Precision Evaluation


## Multi-NCS2 Parallel Inference


## Multi-NCS2 Parallel Inference Optimization


## Experimental Results
### Precision and Single-stick Speed

### Multi-NCS2 Inference Speed


## Reference