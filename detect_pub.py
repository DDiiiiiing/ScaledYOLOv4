#!/usr/bin/env python3

from threading import Thread
from unittest import result

import rospy
from sensor_msgs.msg import Image as msgImage
from std_msgs.msg import String as msgString
from scaledyolo.msg import Yolo as msgYolo
from cv_bridge import CvBridge, CvBridgeError
import sys

import os
import time

import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

class LoadTopic:  # for inference
    def __init__(self, rgb_image_topic, img_size=640):
        self.img_size = img_size
        self.img=None

        self.bridge = CvBridge()
        self.sub_rgb=rospy.Subscriber(rgb_image_topic,msgImage,self.imageRGBCallback)
        self.pub_yolo = rospy.Publisher('/dsr_tray/yolo_result', msgYolo, queue_size=5)

        self.yolo=msgYolo()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        timeout=1
        t1=time.time()
        while self.img is None and time.time()-t1<timeout:
            rospy.logwarn('no image topic')
            rospy.sleep(1)
        if self.img is None:
            rospy.logwarn('timeout!!')
            raise StopIteration

        img0 = self.img
        self.img = None
        img_path = 'image topic'

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0

    def imageRGBCallback(self,Image): 
        try:
            cv_rgbimg=self.bridge.imgmsg_to_cv2(Image,"bgr8")
            self.img=cv_rgbimg
        except CvBridgeError as e:
            print(e)
            self.img=None
            return
        except Exception as e:
            print(e)
            self.img=None
            return

def detect(weights='', imgsz=640, conf_thres=0.4, iou_thres=0.5, dev='', agnostic_nms=False, augment=False):
    # Initialize
    device = select_device(dev) # select default device(device 0)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    if weights == '':
        weight_path=os.path.dirname(os.path.abspath(__file__))+ '/runs/exp11_yolov4-csp-results/weights/' + 'best_yolov4-csp-results.pt'
    
    # Load model
    model = attempt_load(weight_path, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    global dataset

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    with torch.no_grad():
        for path, img, im0s, vid_cap in dataset:
            dataset.yolo.image=dataset.bridge.cv2_to_imgmsg(im0s)
            # cv2.imshow('im0', im0s)
            # cv2.waitKey(0)
            # rospy.logwarn(torch.cuda.memory_allocated()/1024**2)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
            t2 = time_synchronized()
            
            # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for det in pred:  # detections per image
                s, im0 = '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    detection_result = ''
                    for *xyxy, conf, cls in det:
                        # write result
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # detection_result += ('%g ' * 5 + '\n') % (cls, *xywh) # label format
                        detection_result += ('%g ' * 5 + '\n') % (cls, *xyxy) # label format
                        label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2, rect_thickness=1)
                    rospy.loginfo(detection_result)
                    
                    result_str = msgString()
                    result_str.data= detection_result
                    dataset.yolo.string=result_str
                    
                    dataset.pub_yolo.publish(dataset.yolo) #publish original image to find handle
            cv2.imshow('detection', im0)
            cv2.waitKey(1)

    # Print time (inference + NMS)
    # print('%sDone. (%.3fs)' % (s, t2 - t1))
    
    return

def main():
    global dataset
    rgb_image_topic= "/camera/color/image_raw"
    # rgb_image_topic= "/usb_cam/image_raw"
    # rgb_image_topic= "/eye_to_hand_cam/color/image_raw"
    dataset = LoadTopic(rgb_image_topic)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

#####################
global dataset

if __name__=='__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    rospy.loginfo("press q to quit")
    
    th1 = Thread(target=main)
    rospy.sleep(3) # wait until node starts
    th2 = Thread(target=detect, args=('', 416, 0.4, 0.5, '', False, False), daemon=True)

    th1.start()
    th2.start()
    th1.join()
    th2.join()