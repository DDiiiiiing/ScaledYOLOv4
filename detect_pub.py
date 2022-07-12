#!/usr/bin/env python2  

import rospy
import cv2
from sensor_msgs.msg import Image as msg_Image
from std_msgs.msg import String as msg_String
from cv_bridge import CvBridge, CvBridgeError
import sys

import os
import time

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

class DetectYOLO:     
    def __init__(self,rgb_image_topic):
        self.bridge = CvBridge()
        self.sub_rgb=rospy.Subscriber(rgb_image_topic,msg_Image,self.imageRGBCallback)
        self.pub_img=rospy.Publisher('/dsr/tray/yolo_img', msg_Image, queue_size=5)
        self.pub_result=rospy.Publisher('dsr/tray/yolo_result', msg_String, queue_size=5)

    def imageRGBCallback(self,Image):
        try:
            cv_rgbimg=self.bridge.imgmsg_to_cv2(Image,"bgr8")
            img_labeled, result = detect(cv_rgbimg)
            
            img_labeled_resize=cv2.resize(img_labeled, (640, 360))
            cv2.imshow("detected",img_labeled_resize)

            result_str=''
            for r in result:
                cls = r[0]
                [x1, y1, x2, y2] = r[1:]
                str+=str(cls)+' '+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'/'
            
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(img_labeled))
            self.pub_result.publish(result_str)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
            return   

def detect(frame, weights='./runs/best_yolov4-p5-result.pt', imgsz=640, conf_thres=0.4, iou_thres=0.5, dev='', agnostic_nms=False, augment=False):
    # Initialize
    device = select_device(dev) # select default device(device 0)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    dataset = LoadImages(frame, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    for path, img, im0s, vid_cap in dataset:
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
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        result=[]
        # Process detections
        for i, det in enumerate(pred):  # detections per image\
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

                # Write Result
                for *xyxy, conf, cls in det:
                    result.append([cls, *xyxy])
                    label = '%s' % (names[int(cls)])
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))
        
        return im0, result

def main():
    # rgb_image_topic= "/camera/color/image_raw"
    rgb_image_topic= "/eye_to_hand_cam/color/image_raw"
    listner = DetectYOLO(rgb_image_topic)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()
