# -*- coding: utf-8 -*-
__author__ = '翁飞龙'
import cv2
from ModelTraining import Model
import socket
 
if __name__ == '__main__':
    # 加载模型
    model = Model()
    model.load_model(file_path='./Model/me.face.model.h5')
 
    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)
 
    # 捕获指定摄像头的实时视频流
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
 
    # 人脸识别分类器本地存储路径
    cascade_path = "D:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
 
    # 循环检测识别人脸
    while True:
        ret, img = camera.read()  # 读取一帧视频
 
        # 图像灰化，降低计算复杂度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)
 
        # 利用分类器识别出哪个区域为人脸
        fac_gray = cascade.detectMultiScale(gray, 1.1, 5)
        if len(fac_gray) > 0:
            for (x, y, w, h) in fac_gray:
                # 截取脸部图像提交给模型识别这是谁
                image = img[y: y + h, x: x + w]
                faceID = model.face_predict(image)
                print(faceID)
 
                # 如果是“我”
                if faceID == 0:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
 
                    # 文字提示是谁
                    cv2.putText(img, 'wengfeilong',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                    client.connect(('192.168.174.128',6666))
                    print('\n Client is running')
                    client.send(str(faceID).encode('utf-8'))
                    #data = client.recv(1024).decode('utf-8')
                    #client.send("yes".encode("utf-8"))  #响应服务器端发送请求，为防止粘包的产生
                    #print(data)

                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
 
                    # 文字提示是谁
                    cv2.putText(img, 'others',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)
                    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                    client.connect(('192.168.174.128',6666))
                    print('\nClient is running...')
                    client.send(str(faceID).encode('utf-8'))
                    #data = client.recv(1024).decode('utf-8')
                    #client.send("no".encode("utf-8"))  #响应服务器端发送请求，为防止粘包的产生
                    #print(data)
 
        cv2.imshow("camera", img)
 
        # 等待200毫秒看是否有按键输入
        k = cv2.waitKey(200)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
 
    # 释放摄像头并销毁所有窗口
    camera.release()
    cv2.destroyAllWindows()