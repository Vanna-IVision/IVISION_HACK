import cv2

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_parking_boxes(path):
    import xml.etree.ElementTree as ET
    names = ["xmin", "ymin", "xmax", "ymax"]
    data = dict()
    for name in names:
        data[name] = []
    for event, elem in ET.iterparse(path):
        if elem.tag in names:
            data[elem.tag].append(int(elem.text))
    boxes = []
    for i in range(len(data[names[0]])):
        box = []
        for name in names:
            box.append(data[name][i])
        boxes.append(box)

    return boxes


    return data


def reset(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr[i][j] = 0


def draw_result(image,show_value,ind,fill_percentage,thresh,parking_places):
    image = cv2.putText(image, "Free spots: " + str(show_value), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 10, cv2.LINE_AA)
    for id, val in enumerate(fill_percentage[ind]):
        if id < len(parking_places[ind]):
            if val < thresh:
                x0, y0, x1, y1 = parking_places[ind][id]
                color = (200, 10, 200)
                thickness = 10
                image = cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

    return image