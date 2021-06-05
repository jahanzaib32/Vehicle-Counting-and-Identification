import cv2
import pytesseract as tess
from sklearn.cluster import KMeans
from imutils import *
from project.annotations import *
from project.readplate import *

class RecognitionAndDetection():
    def __init__(self, carModel, carbPBTXT, plateModel, platePBTXT):
        self.carPlateNetwork = cv2.dnn.readNetFromTensorflow(carModel, carbPBTXT)
        self.numPlateReader = PlateReader(plateModel, platePBTXT)

    def detectObjects(self, img, car=True, plate=True, threshold=0.4):
        img = resize(img, width=1000)
        rows, cols, channels = img.shape
        annotations = SceneAnnotations(img)
        
        self.carPlateNetwork.setInput(cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300)))
        networkOutput = self.carPlateNetwork.forward()

        for detection in networkOutput[0,0]:
            score = float(detection[2])

            if score > threshold:

                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows

                #handel car detection
                if (car == True) and (detection[1] == 1.0):
                    #do car stuff
                    annotations.addCar(left, top, right, bottom)

                if (plate == True) and (detection[1] == 2.0):
                    num = self.numPlateReader.readNumPlate(img[int(top):int(bottom), int(left):int(right)])
                    annotations.addPlate(left, top, right, bottom, num)
        
        return annotations
    
    def showDetections(self, annotationClass, tracker, count):
        img = annotationClass.getImage()
        annotations = annotationClass.getAnnotations()

        for car in annotations["cars"]:
            cv2.rectangle(  img, 
                            (int(car["left"]), int(car["top"])),
                            (int(car["right"]), int(car["bottom"])),
                            (0, 0, 255), 
                            thickness=2
                        )

        for plate in annotations["plates"]:
            vehicleNum = tracker.getIDFromNumPlate(plate["numplate"])
            cv2.putText(img,
                str(vehicleNum) + ": " + str(tracker.getNumPlateFromID(vehicleNum)), 
                (int(plate["left"]+20), int(plate["top"]-10)), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, 
                (0, 255, 0), thickness=3
            )

            cv2.rectangle(img, 
                (int(plate["left"]), int(plate["top"])), 
                (int(plate["right"]), int(plate["bottom"])), 
                (0, 0, 255), thickness=2
            )

        cv2.imshow("Output", img)
        cv2.imwrite('output/' + str(count) + '.jpg', img)