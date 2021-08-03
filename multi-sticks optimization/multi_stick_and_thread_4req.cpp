#include <map>
#include <queue>
#include <mutex>
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <thread>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <ext_list.hpp>
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>

using namespace std ;
using namespace InferenceEngine ;
using namespace cv ;

#define NCS_MAX_NUM 10              //计算棒最大数量

const int camera_width  = 1280 ;    //摄像头宽度
const int camera_height = 720 ;     //摄像头高度
const int camera_fps = 60 ;         //摄像头帧率
const string xml_file = "/home/jmj/project/NCS_optimization/lrmodels/FP16/frozen_inference_graph.xml" ;
const string bin_file = "/home/jmj/project/NCS_optimization/lrmodels/FP16/frozen_inference_graph.bin" ;
bool flag = true ;
int ncsnum ;                            //计算棒数量
queue <Mat> frame_que ;                 //图像队列
queue <const float * > result_que ;     //结果队列
ostringstream detecfps ;
ostringstream reappearfps ;
ostringstream ncs_number ;
ostringstream request_number ;
mutex mtx1,mtx2 ;                             //全局互斥锁


//初始化标签
void label_init(map<int,string> &labels){
    fstream inf ;
    inf.open("/home/jmj/project/NCS_optimization/labels.txt") ;
    int key ;
    string value ;
    while(inf>>key>>value){
        labels.insert(pair<int,string>(key,value));
    }
    inf.close() ;
    return ;
}

//输出结果处理
Mat Detection_Output(const Mat frame, const float *result, map<int,string> &labels ){
    Mat image = frame ;

    if (result == NULL){
        return image ;
    }

    else {
        for (int i=0; i < 100 ; i++){
            float image_id =result[i * 7 + 0] ;
            if(image_id < 0){
                break ;
            }

            float confidence = result[i * 7 + 2] ;
            auto label = static_cast<int>(result[i * 7 + 1]) ;
            float xmin = result[i * 7 + 3] * camera_width ;
            float ymin = result[i * 7 + 4] * camera_height ;
            float xmax = result[i * 7 + 5] * camera_width ;
            float ymax = result[i * 7 + 6] * camera_height ;

            if(confidence > 0.5){
                ostringstream conf ;
                conf<<":"<<fixed<<setprecision(3)<<confidence ;
                putText(image, labels[label]+conf.str(), Point2f(xmin,ymin-5),FONT_HERSHEY_COMPLEX_SMALL,1,Scalar(0,0,255)) ;
                rectangle(image,Point2f(xmin,ymin),Point2f(xmax,ymax),Scalar(0,0,255)) ;
            }
        }

        putText(image, reappearfps.str(), Point2f(10,25),FONT_HERSHEY_TRIPLEX,0.8,Scalar(255,0,0)) ;
        putText(image, detecfps.str(), Point2f(10,50),FONT_HERSHEY_TRIPLEX,0.8,Scalar(255,0,0)) ;
        putText(image, ncs_number.str(), Point2f(10,75),FONT_HERSHEY_TRIPLEX,0.8,Scalar(255,0,0)) ;
        putText(image, request_number.str(), Point2f(10,100),FONT_HERSHEY_TRIPLEX,0.8,Scalar(255,0,0)) ;

        return image ;
    }

}

//多线程1：摄像头
void camera(){
    //初始化标签
    map<int,string> labels ;
    label_init(labels) ;
    
    //初始化
    typedef chrono::duration<double, ratio<1,1000>> ms ;
    Mat frame ;
    Mat image ;
    const float *res = NULL ;
    const float *lastres = NULL ;
    int detecnt = 0 ;
    int imgcnt = 0 ;
    double imgfps = 0 ;
    double detectime = 0 ;
    ncs_number<<"NCS_number: "<<ncsnum ;
    request_number<<"eachNCS request number: 4" ;

    //初始化摄像头
    VideoCapture cap ;
    cap.open(4) ;
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G')) ;
    cap.set(CAP_PROP_FRAME_WIDTH, camera_width) ;
    cap.set(CAP_PROP_FRAME_HEIGHT, camera_height) ;
    cap.set(CAP_PROP_FPS, camera_fps) ;

    while(flag) {
        auto t0 = chrono::high_resolution_clock::now() ;
        cap >> frame ;

        mtx1.lock() ;                //上锁
        if(frame_que.size()>10) {
            frame_que.pop() ;

        }
        frame_que.push(frame.clone()) ;
        mtx1.unlock() ;              //解除锁
        
        if (result_que.empty()){
            image = Detection_Output(frame, lastres, labels) ;
        }
        else {
            mtx2.lock() ;
            res = result_que.front() ;
            result_que.pop() ;
            mtx2.unlock() ;
            detecnt ++ ;
            image = Detection_Output(frame, res, labels) ;
            lastres = res ;
        }

        imshow("Result", image) ;
        
        imgcnt ++ ;
        if (imgcnt >= 30){
            detecfps.str("") ;
            reappearfps.str("") ;
            detecfps<<"Inference: "<<fixed<<setprecision(3)<<(detecnt*1000.f)/detectime<<"FPS" ;
            reappearfps<<"Reappear: "<<fixed<<setprecision(3)<<imgfps/imgcnt<<"FPS" ;
            imgcnt = 0 ;
            detecnt = 0 ;
            imgfps = 0 ;
            detectime = 0 ;
        }
        auto t1 = chrono::high_resolution_clock::now() ;
        ms total_time = chrono::duration_cast<ms>(t1-t0) ;
        imgfps += 1000.f/total_time.count() ;
        detectime += total_time.count() ;

        if(waitKey(1) == 0x71){
            lock_guard<mutex> lk(mutex) ;
            flag = false ;
            break ;
        }
    }

    destroyAllWindows() ;

    usleep(100) ;

    return ;
}

//多线程2：推断
void inference(Core ie , string device , int ncsid){

    Mat curr_frame ;
    Mat next_frame ;
    Mat prev_frame ;
    Mat four_frame ;
    bool isFirstFrame = true ;
    bool isSecondeFrame = true ;
    bool isThirdFrame = true ;

    //加载网络
    CNNNetReader netReader ;
    netReader.ReadNetwork(xml_file) ;
    netReader.ReadWeights(bin_file) ;
    CNNNetwork network = netReader.getNetwork() ;

    //配置输入blobs
    InputsDataMap InputInfo(network.getInputsInfo()) ;
    string imageInputName ;
    for (const auto & inputInfoItem : InputInfo){
        imageInputName = inputInfoItem.first ;
        inputInfoItem.second->setPrecision(Precision::U8) ;
        inputInfoItem.second->getInputData()->setLayout(Layout::NCHW) ;
    }

    //配置输出blobs
    OutputsDataMap outputInfo(network.getOutputsInfo()) ;
    DataPtr& output = outputInfo.begin()->second ;
    auto outputName = outputInfo.begin()->first ;
    output->setPrecision(Precision::FP32) ;

    //加载插件
    ExecutableNetwork exctnet = ie.LoadNetwork(network, device) ;

    //设置推断请求
    InferRequest::Ptr infer_request_curr = exctnet.CreateInferRequestPtr() ;
    InferRequest::Ptr infer_request_next = exctnet.CreateInferRequestPtr() ;
    InferRequest::Ptr infer_request_prev = exctnet.CreateInferRequestPtr() ;
    InferRequest::Ptr infer_request_four = exctnet.CreateInferRequestPtr() ;

    while(flag){
        if (frame_que.size() < ncsnum){
            usleep(10) ;
            continue ;
        }
        mtx1.lock() ;
        four_frame = frame_que.front() ;
        frame_que.pop() ;
        mtx1.unlock() ;

        if(isFirstFrame){
            prev_frame = four_frame ;
            Blob::Ptr frameBlob = infer_request_prev->GetBlob(imageInputName) ;
            matU8ToBlob<uint8_t>(prev_frame,frameBlob) ;
            infer_request_prev->StartAsync() ;
            isFirstFrame = false ;
        }
        else if(isSecondeFrame){
            curr_frame = four_frame ;
            Blob::Ptr frameBlob = infer_request_curr->GetBlob(imageInputName) ;
            matU8ToBlob<uint8_t>(curr_frame,frameBlob) ;
            infer_request_curr->StartAsync() ;
            isSecondeFrame = false ;
        }
        else if(isThirdFrame){
            next_frame = four_frame ;
            Blob::Ptr frameBlob = infer_request_next->GetBlob(imageInputName) ;
            matU8ToBlob<uint8_t>(next_frame,frameBlob) ;
            infer_request_next->StartAsync() ;
            isThirdFrame = false ;
        }
        else {
            Blob::Ptr frameBlob = infer_request_four->GetBlob(imageInputName) ;
            matU8ToBlob<uint8_t>(four_frame,frameBlob) ;
            infer_request_four->StartAsync() ;
            if(OK==infer_request_prev->Wait(IInferRequest::WaitMode::RESULT_READY)){
                const float *detection = infer_request_prev->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>() ;
                mtx2.lock() ;
                result_que.push(detection) ;
                mtx2.unlock() ;
            }

            infer_request_prev.swap(infer_request_curr) ;
            infer_request_curr.swap(infer_request_next) ;
            infer_request_next.swap(infer_request_four) ;
        }
    
    }

    usleep(100 * ncsid) ;       //每个线程在不同时刻返回

    return ;
}


int main(){

    string ncsdevice[NCS_MAX_NUM] ;
    Core ie ;
    vector<thread> vecthread ;

    vector<string> myriadDevices = ie.GetMetric("MYRIAD", Metrics::METRIC_AVAILABLE_DEVICES) ;
    ncsnum = myriadDevices.size() ;
    for (int i=0; i < ncsnum; i++){
        ncsdevice[i] = string("MYRIAD.") + myriadDevices[i] ;
    }

    //添加多棒线程
    for (int i=0; i < ncsnum; i++){
        vecthread.push_back(thread(inference, ie, ncsdevice[i], i+2)) ;
    }
    //添加摄像头线程
    vecthread.push_back(thread(camera )) ;


    for (int i=0; i < vecthread.size(); i++){
        vecthread[i].join() ;
    }

    cout<<"Finished!"<<endl ;

    return 0;
}