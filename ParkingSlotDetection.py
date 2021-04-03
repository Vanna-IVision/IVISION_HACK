from imageai.Detection import ObjectDetection
import os
from time import time
import cv2
from utils import *
from keras.models import load_model
import copy
import numpy as np
import json

roi_1_coords = (442, 478, 567, 237)
roi_2_coords = (914, 540, 662, 260)
def_rois = [roi_1_coords, roi_2_coords]

execution_path = os.getcwd()
labels_camera1_path = os.path.join(execution_path,"labels","camera_1")


class ParkingSlotDetection:
    def __init__(self, labels_path=labels_camera1_path, thresh=0.31, rois=def_rois,v1=False,camera_id=None):
        self.camera_id = camera_id
        self.thresh = thresh
        self.count = 0
        execution_path = os.getcwd()
        self.v1 = v1
        if v1:
            self.detector = ObjectDetection()
            self.detector.setModelTypeAsRetinaNet()
            self.detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
            self.detector.loadModel()

            self.custom_objects = self.detector.CustomObjects(car=True, bus=True, truck=True)
        else:
            IMAGE_WIDTH = 110
            IMAGE_HEIGHT = 110
            self.dim =(IMAGE_WIDTH, IMAGE_HEIGHT)
            self.detector = load_model(os.path.join(execution_path,'model_ivision_4.h5'))

        self.load_labels(labels_path)

        self.rois = rois
        # self.parking_places = []
        # for file in os.listdir(labels_path):
        #     if file.endswith(".xml"):
        #         self.parking_places.append(get_parking_boxes(os.path.join(labels_path, file)))
        #         # print(file," ",len(parking_places[len(parking_places)-1]))
        # self.fill_percentage = []
        # temp = []
        # for i in range(len(self.parking_places[0])):
        #     temp.append(0.0)
        # for i in range(len(self.parking_places)):
        #     self.fill_percentage.append(copy.deepcopy(temp))

    def load_labels(self, labels_path):
        self.parking_places = []

        for file in os.listdir(labels_path):
            if file.endswith(".xml"):
                self.parking_places.append(get_parking_boxes(os.path.join(labels_path, file)))
                print(file," ",len(self.parking_places[len(self.parking_places)-1]))

        self.fill_percentage = []

        temp = []

        max_val = max(len(i) for i in self.parking_places)
        #print(max_val)
        for i in range(max_val):
            temp.append(0.0)
        for i in range(len(self.parking_places)):
            self.fill_percentage.append(copy.deepcopy(temp))

    def load_rois(self, rois):
        self.rois = rois

    def processing(self, image):
        reset(self.fill_percentage)
        for (x, y, w, h) in self.rois:
            roi = image[y:y + h, x:x + w]

            image_copy, detections = self.detector.detectObjectsFromImage(custom_objects=self.custom_objects,
                                                                          input_type="array",
                                                                          input_image=roi,
                                                                          output_type="array",
                                                                          minimum_percentage_probability=30)

            for eachObject in detections:
                box = eachObject["box_points"]
                box[0] += x
                box[1] += y
                box[2] += x
                box[3] += y
                # x0, y0, x1, y1 = box
                # color = (255, 255, 10)
                # thickness = 2
                # image = cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
                for i in range(len(self.parking_places)):
                    #print("len:",len(self.parking_places[i]))
                    for j in range(len(self.parking_places[i])):
                        #print(i," ",j)
                        self.fill_percentage[i][j] += bb_intersection_over_union(box, self.parking_places[i][j])

        # for i in self.fill_percentage:
        #     print(i)
        counts = []
        for i in range(len(self.fill_percentage)):
            count = 0
            for index,j in enumerate(self.fill_percentage[i]):
                if index < len(self.parking_places[i]):
                    if j < self.thresh: count += 1
            counts.append(count)

        copy_img = image.copy()
        ind = np.argmin(counts)
        show_value = counts[ind]
        show_value = str(show_value) + " (min)"
        min_image = draw_result(image, show_value, ind, fill_percentage=self.fill_percentage, thresh=self.thresh,
                                parking_places=self.parking_places)
        print("min:", self.fill_percentage[ind])

        ind = np.argmax(counts)
        show_value = counts[ind]
        show_value = str(show_value) + " (max)"
        max_image = draw_result(copy_img, show_value, ind, fill_percentage=self.fill_percentage, thresh=self.thresh,
                                parking_places=self.parking_places)
        print("max:", self.fill_percentage[ind])

        numpy_horizontal_concat = np.concatenate((min_image, max_image), axis=0)

        return numpy_horizontal_concat


    def process_v2(self,image):
        self.count+=1
        t = time()
        pred = []
        for part in self.parking_places:
            batch = []
            for place in part:
                x, y, w, h = place
                w = w - x
                h = h - y
                roi = image[y:y + h, x:x + w]
                roi = cv2.resize(roi, self.dim, interpolation=cv2.INTER_CUBIC)
                batch.append(roi)

            batch = np.stack(batch)
            # print(len(batch))
            y_pred = self.detector.predict(batch, batch_size=len(batch), verbose=0)
            y_pred_bool = np.argmax(y_pred, axis=1)
            pred.append(y_pred_bool)

        s = []
        for i in pred:
            s.append(np.sum(i))

        max_image = image.copy()
        min_image = image.copy()

        ind = np.argmax(s)
        show_value = s[ind]
        show_value = str(show_value) + " (max)"
        max_image = draw_result_v2(max_image,ind,pred,self.parking_places,show_value)
        max_image = cv2.resize(max_image,(1080,720))
        if(self.count%30==0):
            cv2.imwrite(os.path.join(os.getcwd(), "image_" + str(self.camera_id) + ".jpg"), max_image)
            with open(os.path.join(os.getcwd(),'data.json'), 'w') as f:
                json.dump({"value":str(s[ind])}, f)


        ind = np.argmin(s)
        show_value = s[ind]
        show_value = str(show_value) + " (min)"
        min_image = draw_result_v2(min_image,ind,pred,self.parking_places,show_value)
        min_image = cv2.resize(min_image, (1080, 720))
        numpy_horizontal_concat = np.concatenate((min_image, max_image), axis=0)
        print(1./(time()-t))

      #  time.sleep(0.1)

        return numpy_horizontal_concat