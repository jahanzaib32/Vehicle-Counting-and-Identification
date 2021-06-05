import cv2
import numpy as np
import time
import os
from copy import deepcopy
from PIL import Image
import pytesseract as tess
import threading
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import fyp.pointTrack


THREADING = False
DEBUG = False
PRINT_NUM = True

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


lock = threading.Lock()
tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph1.pb', 'output.pbtxt')
tensorflowNet_plate = cv2.dnn.readNetFromTensorflow('frozen_inference_graph_plate1.pb', 'output_plate.pbtxt')
tess.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'




def findDomColor(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1],3))

        clt = KMeans(n_clusters=1)
        clt.fit(img)
        return clt.cluster_centers_[0]
    except:
        return (0, 0, 0)

def writeFile(dataArray, fileName):
    with lock:
        with open(fileName, mode='a') as data_file:
            file_writer= csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(dataArray)

def preprocess_new(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue

def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

def readNumPlate(img, tensorflowNet, debug=False):
    plate_num = ""
    rows, cols, channels = img.shape
    
    with lock:
        tensorflowNet.setInput(cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300)))
        networkOutput = tensorflowNet.forward()

    for detection in networkOutput[0,0]:
        score = float(detection[2])

        if score > 0.6:
            
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            numplate = img[int(top):int(bottom), int(left):int(right)]
            numplate = image_resize(numplate, width=250)
            plate_num = plate_num + str(read_plate(numplate, preprocess=True, debug=debug))
    if THREADING:
        writeFile([plate_num.replace("\n", "")], "numbers.csv")
    
    return plate_num

def preprocess_plate(img, read=False):
    org_img = img.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.Canny(img, 100, 200)
    _, img = cv2.threshold(img,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel3 = np.ones((4,4),np.uint8)
    kernel_long_height = np.ones((4,8),np.uint8)
    img = cv2.dilate(img, kernel_long_height, iterations = 1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)

    return img

def read_plate(img, preprocess=True, debug=False):
    img_cp = img.copy()
    if preprocess:
        img, _ = preprocess_new(img)
    if debug:
        cv2.imshow("Preprocessed", img)
        cv2.waitKey()
    test_image = Image.fromarray(img)
    text = tess.image_to_string(test_image, lang='eng', config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 13") # --psm 13
    
    return text.replace('\n', '')

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def detectAndShow(folder, pic, tensorflowNet):
    img = cv2.imread(os.path.join(folder, str(pic) + ".jpg")) #+ ".jpg"
    img = image_resize(img, width=1000)
    rect_img = img.copy()
    rows, cols, channels = img.shape

    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300)))
    networkOutput = tensorflowNet.forward()

    for detection in networkOutput[0,0]:
        score = float(detection[2])

        if score > 0.4:

            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            
            if detection[1] == 1.0:
               ''' vehicle = img[int(top):int(bottom), int(left):int(right)]
                vehicle = image_resize(vehicle, width=90)
                color = findDomColor(vehicle)
                #print(color)
                cv2.putText(rect_img,
                                str("|||"), 
                                (int(left+20), int(top-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, 
                                (color[0], color[1], color[2]), thickness=40
                            )'''
            elif detection[1] == 2.0:
                numplate = img[int(top):int(bottom), int(left):int(right)]
                numplate = image_resize(numplate, width=200)
                if THREADING:
                    number = threading.Thread(target=readNumPlate, args=(numplate, tensorflowNet_plate, False), daemon=True)
                    number.start()
                else:
                    number = readNumPlate(numplate, tensorflowNet_plate, debug=DEBUG)
                    if PRINT_NUM:
                        print(number)
                    cv2.putText(rect_img,
                                    str(number), 
                                    (int(left+20), int(top-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, 
                                    (0, 0, 255), thickness=4
                                )

            cv2.rectangle(rect_img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    rect_img = image_resize(rect_img, width=600) #image_resize(rect_img, width=1000)
    cv2.imshow("Output", rect_img)
    
def detectAndSave(folder, pic, tensorflowNet):
    img = cv2.imread(os.path.join(folder, str(pic) + ".jpg")) #+ ".jpg"
    img = image_resize(img, width=1000)
    rect_img = img.copy()
    rows, cols, channels = img.shape

    with lock:
        tensorflowNet.setInput(cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300)))
        networkOutput = tensorflowNet.forward()

    for detection in networkOutput[0,0]:
        score = float(detection[2])

        if score > 0.4:

            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            if detection[1] == 2.0:
                numplate = img[int(top):int(bottom), int(left):int(right)]
                numplate = image_resize(numplate, width=200)
                if THREADING:
                    number = threading.Thread(target=readNumPlate, args=(numplate, tensorflowNet_plate, False), daemon=True)
                    number.start()
                else:
                    number = readNumPlate(numplate, tensorflowNet_plate, debug=DEBUG)
                    if PRINT_NUM:
                        print(number)
                    cv2.putText(rect_img,
                                    str(number), 
                                    (int(left+20), int(top-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, 
                                    (0, 0, 255), thickness=4
                                )

            cv2.rectangle(rect_img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    #rect_img = img #image_resize(rect_img, width=1000)
    cv2.imwrite("output1/" + str(pic) + ".jpg", rect_img)