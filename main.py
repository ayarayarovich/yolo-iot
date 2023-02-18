import cv2
import numpy as np

import time
import os

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def main():
    # the neural network configuration
    config_path = "cfg/yolov3.cfg"
    # the YOLO net weights file
    weights_path = "weights/yolov3.weights"

    # loading all the class labels (objects)
    labels = open("data/coco.names").read().strip().split("\n")
    # generating colors for each object for later plotting
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # load the YOLO network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # path_name = "images/city_scene.jpg"
    cap = cv2.VideoCapture(1)
    while True:
        ret, image = cap.read()
        # file_name = os.path.basename(path_name)
        # filename, ext = file_name.split(".")

        h, w = image.shape[:2]
        # create 4D blob
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # sets the blob as the input of the network
        net.setInput(blob)

        # get all the layer names
        ln = net.getLayerNames()
        try:
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        except IndexError:
            # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
            ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        # feed forward (inference) and get the network output
        layer_outputs = net.forward(ln)

        boxes, confidences, class_ids = [], [], []

        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the object detections
            for detection in output:
                # extract the class id (label) and confidence (as a probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # discard weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # perform the non maximum suppression given the scores defined before
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        font_scale = 1
        thickness = 1


        def include_peoples():
            count = 0
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    if not(class_ids[i]):
                        # # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        count += 1
            return count > 0
        
        def include_peoples_fast():
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    if not(class_ids[i]):
                        # # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        return True
            return False

        if (include_peoples()):
            print("Found!")
        else:
            print("Not found!")

        cv2.imshow("image", image)
        if cv2.waitKey(1) == ord("q"):
            break

        # cv2.imwrite(filename + "_yolo3." + ext, image)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()