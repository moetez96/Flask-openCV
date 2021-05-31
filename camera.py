# Modified by smartbuilds.io
# Date: 27.09.20
# Desc: This scrtipt script..

import cv2
import imutils
import time
import numpy as np
import sys
import face_recognition
import os
from datetime import datetime

cascPath = sys.argv[0]
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

path = 'C:/Users/HP/Desktop/camera stream/static/pictures'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

thres = 0.5  # Threshold to detect object
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress

classNamesObject = []
classFile = 'C:/Users/HP/Desktop/camera stream/basic/Object_Detection_Files/coco.names'
with open(classFile, 'rt') as f:
    classNamesObject = f.read().splitlines()
print(classNamesObject)

font = cv2.FONT_HERSHEY_PLAIN
# font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNamesObject), 3))

configPath = 'C:/Users/HP/Desktop/camera stream/basic/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'C:/Users/HP/Desktop/camera stream/basic/Object_Detection_Files/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('C:/Users/HP/Desktop/camera stream/static/Saved.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')


class VideoCamera(object):
    def __init__(self, flip=False):
        self.vs = cv2.VideoCapture(0)
        self.flip = flip
        time.sleep(2.0)


    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame_object(self):

        while self.vs.isOpened():
            success, frame = self.vs.read()
            classIds, confs, bbox = net.detect(frame, confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            # print(type(confs[0]))
            # print(confs)

            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
            if len(classIds) != 0:
                for i in indices:
                    i = i[0]
                    box = bbox[i]
                    confidence = str(round(confs[i], 2))
                    color = Colors[classIds[i][0] - 1]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
                    cv2.putText(frame, classNamesObject[classIds[i][0] - 1] + " " + confidence, (x + 10, y + 20),
                                font, 1, color, 2)
            #             cv2.putText(img,str(round(confidence,2)),(box[0]+100,box[1]+30),
            #                         font,1,colors[classId-1],2)

            ret1, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

    def get_frame_normal(self):
        while self.vs.isOpened():
            ret, frame = self.vs.read()

            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret1, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

    def get_frame(self):
        while self.vs.isOpened():
            ret, frame = self.vs.read()

            if not ret:
                break

            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if faceDis[matchIndex] < 0.50:
                    name = classNames[matchIndex].upper()
                    markAttendance(name)
                else:
                    name = 'Unknown'
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            ret1, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

    """
        def get_frame(self):
        i = 0
        while self.vs.isOpened():
            ret, frame = self.vs.read()

            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.imwrite('kang' + str(i) + '.jpg', frame)
            ret1, jpeg = cv2.imencode('.jpg', frame)
            i += 1
            return jpeg.tobytes()
"""
