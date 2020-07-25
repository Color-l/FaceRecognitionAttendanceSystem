# -*- coding: utf-8 -*-
__author__ = '翁飞龙'
import cv2
import numpy as np
import os

def path():
    name=input('\n enter user name:')
    path="./FaceImageDate/" + str(name)
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)  
        print ( '\n ' + path + ' 创建成功')
    else:
        print ( '\n ' + path+ ' 名称已存在')

    return path
def CatchPICFromVideo(path,catch_num):
    faceCascade = cv2.CascadeClassifier(r'D:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier(r'D:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages\cv2\data\haarcascade_eye.xml')

    # 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_id = input('\n enter user id:')
    print('\n 采集数据前请摘下您的眼镜、口罩等遮蔽物，请保持光线良好 ... ')
    print('\n 正在采集人脸数据，请稍后 ...')

    count = 0

    while True:

        # 从摄像头读取图片

        sucess, img = camera.read()

        # 转为灰度图片

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64)
        )
        # 在检测人脸的基础上检测眼睛
        result = []
        for (x, y, w, h) in faces:
            fac_gray = gray[y: (y+h), x: (x+w)]
            eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 2)

            # 眼睛坐标的换算，将相对位置换成绝对位置
            for (ex, ey, ew, eh) in eyes:
                result.append((x+ex, y+ey, ew, eh))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0),2)
        for (ex, ey, ew, eh) in result:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # 显示捕捉了多少张人脸
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'count:{str(count)}', (x + 30, y + 30), font, 1, (255, 0, 255), 4)

            count += 1

            img_name = path + '/' + str(face_id) + '_' + str(count) + '.jpg'
            # 保存图像（路径不能包含中文）
            #cv2.imwrite(img_name, gray[y: y + h, x: x + w])

            #保存图像
            cv2.imencode('.jpg',  gray[y: y + h, x: x + w])[1].tofile(img_name)

            cv2.imshow('image', img)

        # 保持画面的持续。1ms

        k = cv2.waitKey(1)

        if k == 27:   # 通过esc键退出摄像
            break
        elif count >= catch_num:
            break;
    
    print("\n 人脸信息采集完成")
    # 关闭摄像头
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path=path()
    CatchPICFromVideo(path,100)