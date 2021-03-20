from flask import Flask, render_template, Response
import cv2
from ParkingSlotDetection import ParkingSlotDetection
import os
app = Flask(__name__)

detector = ParkingSlotDetection()
camera1_path = r"G:\ivision\hsOJsEu\camera_1.mp4"
camera3_path = r"G:\ivision\hsOJsEu\camera_3.mp4"
execution_path = os.getcwd()
labels_camera1_path = os.path.join(execution_path,"labels","camera_1")
labels_camera3_path = os.path.join(execution_path,"labels","camera_3")



def gen_frames(path):  # generate frame by frame from camera
    camera = cv2.VideoCapture(path)
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            camera = cv2.VideoCapture(path)
            #break
        else:
            frame = detector.processing(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/camera1')
def video_feed_1():
    #Video streaming route. Put this in the src attribute of an img tag
    roi_1_coords = (442, 478, 567, 237)
    roi_2_coords = (914, 540, 662, 260)
    rois = [roi_1_coords, roi_2_coords]
    detector.thresh = 0.3
    detector.load_labels(labels_camera1_path)
    detector.load_rois(rois)
    return Response(gen_frames(camera1_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera3')
def video_feed_3():

    #roi_1_coors = (365, 402, 876, 406)
    roi_1_coords = (393,419,813,352)
    rois = [roi_1_coords]
    detector.thresh = 0.27
    detector.load_labels(labels_camera3_path)
    detector.load_rois(rois)
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(camera3_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)
    #app.run(debug=False,threaded = True)
    #app.run(host="192.168.2.64")