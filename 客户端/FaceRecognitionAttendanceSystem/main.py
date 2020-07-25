import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import FaceRecognitionAttendanceSystem
# coding:utf-8
def click_ImageAcquisition():
    pass
def click_FaceDateSet():
    pass
def click_ModelTraining():
    pass
def click_FaceRecognition():
    pass
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = FaceRecognitionAttendanceSystem.Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.ImageAcquisition.clicked.connect(click_ImageAcquisition)#绑定成员录入按钮
    ui.FaceDateSet.clicked.connect(click_FaceDateSet)#绑定数据处理按钮
    ui.ModelTraining.clicked.connect(click_ModelTraining)#绑定模型训练按钮
    ui.FaceRecognition.clicked.connect(click_FaceRecognition)#绑定人脸识别按钮

    sys.exit(app.exec_())