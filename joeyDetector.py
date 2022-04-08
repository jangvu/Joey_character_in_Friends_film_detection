import cv2
import numpy as np
import glob

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()


def dectector_demo(video = False):
    if video == True:
        cap = cv2.VideoCapture('joey_test.mp4')
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(100, 3))

        while True:
            _, img = cap.read()
            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
            print('this is the index')
            print(indexes)
         
            if len(indexes)>0:
                if len(indexes) == 1:
                    x, y, w, h = boxes[0]
                    label = str(classes[class_ids[0]])
                    confidence = str(round(confidences[0],2))
                    color = colors[0]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)    
                else:
                    for i in indexes.flatten():
                        temp = confidences[i]
                        if temp == np.max(confidences):
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            confidence = str(round(confidences[i],2))
                            color = colors[i]
                            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)    
                            
                                    
            cv2.imshow('Image', img)
            key = cv2.waitKey(1)
            if key==27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:  
        #img_path = glob.glob('//Users/jang/Desktop/Hadoop_practice/joey_dectector/*.jpeg')
        path = '//Users/jang/Desktop/Hadoop_practice/joey_dectector/img2.jpeg'
        img_path = [path]
        for image in img_path:
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(100, 3))
            img = cv2.imread(image)
            height, width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            layer_names = net.getLayerNames()
            outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
            net.setInput(blob)
            outputs = net.forward(outputlayers)
            boxes = []
            confidences = []
            class_ids = []
            count = 0
            for output in outputs:
                for detection in output:
        # Each bounding box is represented by 6 numbers (pc, bx, by, bh, bw, c)
        # Next is the number of classes, only 1 class->length of the each output is 6
        # The shape of detection kernel is 1 x 1 x (B x (5 + C)). 
        # Here B is the number of bounding boxes a cell on the feature map can predict, 
        # '5' is for the 4 bounding box attributes and one object confidence and 
        # C is the no. of classes.

                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    count += 1
                    if confidence > 0.2:
                        print(float(confidence))
                        print(count)
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
            if len(indexes)>0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)     
            
            cv2.imshow('window',  img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
dectector_demo(video=True)

    