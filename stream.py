from flask import Flask, render_template, Response
import cv2
from ParkingSlotDetection import ParkingSlotDetection
import os
from imutils.video import WebcamVideoStream
from utils import get_free_from_json
import codecs
app = Flask(__name__)
execution_path = os.getcwd()
print(execution_path)
detector = ParkingSlotDetection(v1=False)

# camera1_path = os.path.join(execution_path,"camera_1.mp4")
camera1_path = "rtsp://93.190.206.140:8554/lenina"
# camera3_path = os.path.join(execution_path,"camera_3.mp4")
camera3_path = "rtsp://93.190.206.140:8554/anohina"

labels_camera1_path = os.path.join(execution_path, "labels", "camera_1")
labels_camera3_path = os.path.join(execution_path, "labels", "camera_3")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


def gen_frames(path):  # generate frame by frame from camera
    camera = WebcamVideoStream(path).start()
    # camera = cv2.VideoCapture(path,cv2.CAP_FFMPEG)
    # camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # camera.set(cv2.CAP_PROP_FPS,1)
    while True:
        # Capture frame-by-frame
        success = True
        frame = camera.read()
        # success, frame = camera.read()  # read the camera frame
        if not success:
            # camera = cv2.VideoCapture(path)
            pass
            # break
        else:
            if detector.v1:
                frame = detector.processing(frame)
            else:
                frame = detector.process_v2(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/camera1')
def video_feed_1():
    # Video streaming route. Put this in the src attribute of an img tag
    roi_1_coords = (442, 478, 567, 237)
    roi_2_coords = (914, 540, 662, 260)
    rois = [roi_1_coords, roi_2_coords]
    detector.thresh = 0.3
    detector.load_labels(labels_camera1_path)
    detector.load_rois(rois)
    detector.camera_id = 1
    return Response(gen_frames(camera1_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera3')
def video_feed_3():
    # roi_1_coors = (365, 402, 876, 406)
    roi_1_coords = (393, 419, 813, 352)
    rois = [roi_1_coords]
    detector.thresh = 0.27
    detector.load_labels(labels_camera3_path)
    detector.load_rois(rois)
    detector.camera_id = 3
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(camera3_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    with codecs.open(os.path.join(os.getcwd(),'templates\index.html'), 'r', 'utf-8') as f:
        all = f.read()
        num = get_free_from_json()
        all = all.replace("-1", num)
        return all
    # return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
    # app.run(debug=False,threaded = True)
    # app.run(host="192.168.2.64")
