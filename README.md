# FaceRecognitionAttendanceSystem
基于Python+opencv+keras+numpy+sklearn的人脸识别门禁系统

欢迎访问个人博客：https://flblog.top

项目地址：https://flblog.top/14.html

FaceRecognitionAttendanceSystem
Python+opencv+keras+numpy+sklearn的人脸识别门禁系统
本项目为实习期间做的一款基于opencv的人脸识别门禁系统，作者某双非二本院校，写于2020/7/23。


作者	翁飞龙
QQ交流群	692695467(点击跳转)
博客地址	https://www.flblog.top

使用环境
windows/Linux,支持Python3.6以上版本和GCC的编辑器

开发环境：Microsoft Visio Studio 2019（Python3.6）

准备材料

1、python环境所需要的包：numpyen、sorflow、keras、opencv、scikit-learn
2、Ubuntu虚拟机
3、M0_smartHome仿真软件
4、串口助手
5、vspd虚拟端口配置工具


注：本文用到的所有工具和源码均在文章末提供下载


一、项目结构

1、客户端




main.py：主界面启动文件（暂未完成）
ImageAcquisition.py：图像采集模块
FaceDateSet.py：图像数据处理模块
ModelTraining.py：模型训练模块
FaceRecognition.py：人脸识别模块
FaceImageDate：文件夹存放采集得到的灰度图片
Model：文件夹存放训练好的人脸模型


2、服务端

server.c服务端源代码
a.out可执行文件


二、运行效果

识别到自己门打开，关闭报警




识别到其他人，关闭门，开始报警




三、源代码(客户端)
1、图像采集
ImageAcquisition.py


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
全选代码复制
2、数据处理
FaceDateSet.py

# -*- coding: utf-8 -*-
__author__ = '翁飞龙'
import os
import numpy as np
import cv2
# 定义图片尺寸
IMAGE_SIZE = 64
 
 
# 按照定义图像大小进行尺度调整
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = 0, 0, 0, 0
    # 获取图像尺寸
    h, w, _ = image.shape
    # 找到图片最长的一边
    longest_edge = max(h, w)
    # 计算短边需要填充多少使其与长边等长
    if h < longest_edge:
        d = longest_edge - h
        top = d // 2
        bottom = d // 2
    elif w < longest_edge:
        d = longest_edge - w
        left = d // 2
        right = d // 2
    else:
        pass
 
    # 设置填充颜色
    BLACK = [0, 0, 0]
    # 对原始图片进行填充操作
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))
 
images, labels = list(), list()
# 读取训练数据
def read_path(path):
    for dir_item in os.listdir(path):
        # 合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path, dir_item))
        # 如果是文件夹，则继续递归调用
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                # print(dir_item)
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                labels.append(path)
    #print(labels)
    return images, labels
 
 
# 从指定路径读取训练数据
def load_dataset(path):
    images, labels = read_path(path)
    # 由于图片是基于矩阵计算的， 将其转为矩阵
    images = np.array(images)
    print(images.shape)
    labels = np.array([0 if label.endswith('wengfeilong') else 1 for label in labels])
    return images, labels
 
 
if __name__ == '__main__':
    images, labels = load_dataset(os.getcwd()+'/FaceImageDate')
    print('\n 读取结束，数据处理完成......')
全选代码复制

3、模型训练
ModelTraining.py


# -*- coding: utf-8 -*-
__author__ = '翁飞龙'
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from FaceDateSet import load_dataset, resize_image, IMAGE_SIZE
import warnings
warnings.filterwarnings('ignore')
 
 
class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 验证集
        # self.valid_images = None
        # self.valid_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 数据加载路径
        self.path_name = path_name
        # 当前库采用的维度顺序
        self.input_shape = None
 
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        # 加载数据集至内存
        images, labels = load_dataset(self.path_name)
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 10))
        #if K.image_dim_ordering() == 'th':
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
 
            # 输出训练集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(test_images.shape[0], 'test samples')
            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)
            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            test_images = test_images.astype('float32')
            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255.0
            test_images /= 255.0
            self.train_images = train_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.test_labels = test_labels
 
 
# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None
 
    # 建立模型
    def build_model(self, dataset, nb_classes=2):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()
 
        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Conv2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))  # 1 2维卷积层
        self.model.add(Activation('relu'))  # 2 激活函数层
 
        self.model.add(Conv2D(32, 3, 3))  # 3 2维卷积层
        self.model.add(Activation('relu'))  # 4 激活函数层
 
        self.model.add(MaxPool2D(pool_size=(2, 2)))  # 5 池化层
        self.model.add(Dropout(0.25))  # 6 Dropout层
 
        self.model.add(Conv2D(64, 3, 3, border_mode='same'))  # 7  2维卷积层
        self.model.add(Activation('relu'))  # 8  激活函数层
 
        self.model.add(Conv2D(64, 3, 3))  # 9  2维卷积层
        self.model.add(Activation('relu'))  # 10 激活函数层
 
        self.model.add(MaxPool2D(pool_size=(2, 2)))  # 11 池化层
        self.model.add(Dropout(0.25))  # 12 Dropout层
 
        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(Dense(512))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(Dense(nb_classes))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果
 
        # 输出模型概况
        self.model.summary()
 
    # 训练模型
    def train(self, dataset, batch_size=20, nb_epoch=100, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作
 
        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.test_images, dataset.test_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转
 
            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)
 
            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.test_images, dataset.test_labels))
 
    MODEL_PATH = './Model/face.model.h5'
 
    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
 
    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
 
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print(f'{self.model.metrics_names[1]}:{score[1] * 100}%')
 
    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        #if K.image_dim_ordering() == 'th' 
        if K.image_data_format() == 'channels_first'and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        #elif K.image_dim_ordering() == 'tf' 
        elif K.image_data_format() == 'channels_last'and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
 
            # 浮点并归一化
        image = image.astype('float32')
        image /= 255.0
 
        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)
        print('result:', result)
 
        # 给出类别预测：0或者1
        result = self.model.predict_classes(image)
 
        # 返回类别预测结果
        return result[0]
 
 
if __name__ == '__main__':
    dataset = Dataset('./FaceImageDate/')
    dataset.load()
 
    # 训练模型
    model = Model()
    model.build_model(dataset)
    # 测试训练函数的代码
    model.train(dataset)
    model.save_model(file_path='./Model/me.face.model.h5')
    # 评估模型
    model = Model()
    model.load_model(file_path='./Model/me.face.model.h5')
    model.evaluate(dataset)
全选代码复制

4、人脸识别
FaceRecognition.py


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
全选代码复制

四、服务端
server.c


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>

#define BAUDRATE B115200 ///Baud rate : 115200
#define DEVICE "/dev/ttyS1"//设置端口号
#define FALSE 0
#define TRUE 1
#define _POSIX_SOURCE 1 //POSIX系统兼容
int SerialPort_Send(int i){

	int fd,res;
	struct termios oldtio,newtio;
	
	fd=open(DEVICE,O_RDWR | O_NOCTTY);
	if(fd<0){
		perror(DEVICE);
		exit(-1);
	}
	tcgetattr(fd,&oldtio);//保存原来的参数
	bzero(&newtio,sizeof(newtio));
	newtio.c_cflag=BAUDRATE | CS8 | CLOCAL | CREAD | HUPCL;
	newtio.c_iflag=IGNBRK;
	newtio.c_oflag=0;
	newtio.c_lflag=ICANON;
	tcflush(fd,TCIFLUSH);
	tcsetattr(fd,TCSANOW,&newtio);//设置串口参数
	printf("%d\n",i);
	if(i==0){
		char openbuf[255]={0xdd,0x05,0x24,0x00,0x09};
		char closebj[255]={0xdd,0x05,0x24,0x00,0x03};
		write(fd,openbuf,5);
		write(fd,closebj,5);
		close(fd);
	}
	else{
		char closebuf[255]={0xdd,0x05,0x24,0x00,0x0a};
		char baojing[255]={0xdd,0x05,0x24,0x00,0x02};
		write(fd,closebuf,5);
		write(fd,baojing,5);
		close(fd);
	}

}

int main()
{
	int sockfd, new_fd;
	struct sockaddr_in my_addr;
	struct sockaddr_in their_addr;
	int sin_size;
	//建立TCP套接口
	if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
	{
		printf("create socket error");
		perror("socket");
		exit(1);
	}
	//初始化结构体，并绑定6666端口
	my_addr.sin_family = AF_INET;
	my_addr.sin_port = htons(6666);
	my_addr.sin_addr.s_addr = INADDR_ANY;
	bzero(&(my_addr.sin_zero), 8);
	int on;
	on = 1;
	setsockopt( sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on) );
	//绑定套接口
	if (bind(sockfd, (struct sockaddr*)&my_addr, sizeof(struct sockaddr)) == -1)
	{
		perror("bind socket error");
		exit(1);
	}
	//创建监听套接口
	if (listen(sockfd, 10) == -1)
	{
		perror("listen");
		exit(1);
	}
	//等待连接
	while (1)
	{
		sin_size = sizeof(struct sockaddr_in);
		printf("server is run......\n");
		//如果建立连接，将产生一个全新的套接字
		if ((new_fd = accept(sockfd, (struct sockaddr*)&their_addr, &sin_size)) == -1)
		{
			perror("accept");
			exit(1);
		}
		printf("accept success.\n");
		//break;

		//生成一个子进程来完成和客户端的会话，父进程继续监听
		if (!fork())
		{
			printf("create new thred success.\n");
			//读取客户端发来的信息
			int numbytes;
			char buff[1024];
			memset(buff, 0, 1024);
			if ((numbytes = recv(new_fd, buff, sizeof(buff), 0)) == -1)
			{
				perror("recv");
				exit(1);
			}
			printf("%s\n", buff);
			printf("--------------------------------------------------------\n\n");
			int i=(strcmp(buff,"0"));
			SerialPort_Send(i);
			/*if(i==0)
			{	
				char success[]="success";
				if (send(new_fd, success, strlen(success), 0) == -1)
					perror("send");
			}
			else{
				char failed[]="failed";
				if (send(new_fd, failed, strlen(failed), 0) == -1)
					perror("send");
			}
			
			close(new_fd);
			exit(0);
			}*/
			close(new_fd);
		}
	}
	close(sockfd);
}
