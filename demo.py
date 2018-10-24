# -*- coding: utf-8 -*-
import sys
import dlib
import cv2
import numpy
import os 


def get_register_img_vector(imgpath):
    print(imgpath)
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)    # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])   # 融合三个颜色通道生成新图片
    dets = detector(img2, 1)
    for index, face in enumerate(dets):
        shape = sp(img, face)
        # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        # 转换为numpy array
        v = numpy.array(face_descriptor)
    return v

def db_register(db_dir):
    db_maps = {}
    for root, dirs, files in os.walk(db_dir):
        for people_dir in dirs:
            db_maps[people_dir] = []
            for subroot, _, imgs in os.walk(root+people_dir):
                for img in imgs:
                    vector = get_register_img_vector(subroot+'/'+img)
                    db_maps[people_dir].append(vector)
    return db_maps

def get_distance(vec1, vec2):
    dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))
    return dist 

def get_target(vec, dataset, thre=0.5):
    mindist = 10000
    minname = "unknown"
    for k, v in dataset.items():
        for each in v:
            dist = get_distance(each, vec)
            if dist < mindist and dist < thre:
                mindist = dist  
                minname = k 
    return mindist, minname


face_rec_model_path = './model/dlib_face_recognition_resnet_model_v1.dat'
predictor_path = './model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector() #获取人脸分类器
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
descriptors = []
candidate = ["liujialing", "guigui", "baijingting", "weidaxun", "wangou"]
# opencv 读取图片，并显示
dataset = db_register('./image/')
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread("./image/test2.jpg", cv2.IMREAD_COLOR)

b, g, r = cv2.split(img)    # 分离三个颜色通道
img2 = cv2.merge([r, g, b])   # 融合三个颜色通道生成新图片

dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果
# enumerate是一个Python的内置方法，用于遍历索引
# index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
# left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
for index, face in enumerate(dets):
    # print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))
    # 2.关键点检测
    shape = sp(img, face)
    # 3.描述子提取，128D向量
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    # 转换为numpy array
    v = numpy.array(face_descriptor)  
    descriptors.append(v)
    mindist, minname=get_target(v, dataset)
    print(mindist, minname)
    # 在图片中标注人脸，并显示
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.putText(img, minname, (left, top), font, 0.8, (0, 255, 0), 1)
    cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("demo", img)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()
