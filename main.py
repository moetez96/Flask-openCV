# Modified by smartbuilds.io
# Date: 27.09.20
# Desc: This web application serves a motion JPEG stream
# main.py
# import the necessary packages
from flask import Flask, render_template, Response, request
from camera import VideoCamera
import time
import threading
import os

pi_camera = VideoCamera(flip=False)

# App Globals (do not edit)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')  # you can customze index.html here


@app.route('/facial.html')
def indexfacial():
    return render_template('facial.html')  # you can customze index.html here


@app.route('/object.html')
def object():
    return render_template('object.html')  # you can customze index.html here


def gen(camera):
    # get camera frame
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gennormal(camera):
    # get camera frame
    while True:
        frame = camera.get_frame_normal()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def genobject(camera):
    # get camera frame
    while True:
        frame = camera.get_frame_object()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/facial')
def video_feed():
    if request.args['type'] == 'normal':
        return Response(gennormal(pi_camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    if request.args['type'] == 'facial':
        return Response(gen(pi_camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    if request.args['type'] == 'object':
        return Response(genobject(pi_camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
