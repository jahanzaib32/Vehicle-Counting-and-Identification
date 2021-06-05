from project.detect import *
from project.annotations import *
from project.pointTrack import *
import cv2
import os

recognition = RecognitionAndDetection("models/car.pb", "models/car.pbtxt", "models/plate.pb", "models/plate.pbtxt")
platesTracker = TrackPoint()
folder = "pics"
for count in range(1, len(os.listdir(folder))):
    img = cv2.imread(folder + "/" + str(count) + ".jpg")
    annotations = recognition.detectObjects(img)
    platesTracker.registerPointsFromAnnotations(annotations)
    recognition.showDetections(annotations, platesTracker, count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey()
cv2.destroyAllWindows()