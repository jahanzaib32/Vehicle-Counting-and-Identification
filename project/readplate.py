import cv2
import pytesseract as tess
import numpy as np
from imutils import *
from PIL import Image

class PlateReader():
    def __init__(self, plateModel, platePBTXT):
        self.plateNetwork = cv2.dnn.readNetFromTensorflow(plateModel, platePBTXT)
        tess.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        self.GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
        self.ADAPTIVE_THRESH_BLOCK_SIZE = 19
        self.ADAPTIVE_THRESH_WEIGHT = 9

    def readNumPlate(self, img):
        img = resize(img, width=200)
        plate_num = ""
        rows, cols, channels = img.shape

        self.plateNetwork.setInput(cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300)))
        networkOutput = self.plateNetwork.forward()

        for detection in networkOutput[0,0]:
            score = float(detection[2])

            if score > 0.6:
                
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows

                numplate = img[int(top):int(bottom), int(left):int(right)]
                #numplate = resize(numplate, width=250)
                plate_num = plate_num + str(self.readText(numplate))
        
        return plate_num

    def readText(self, img, preprocess=True):
        img_cp = img.copy()
        if preprocess:
            img, _ = self.preprocess(img)

        test_image = Image.fromarray(img)
        text = tess.image_to_string(test_image, lang='eng', config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 13") # --psm 13
        
        return text.replace('\n', '')

    def preprocess(self, imgOriginal):
        imgGrayscale = self.extractValue(imgOriginal)

        imgMaxContrastGrayscale = self.maximizeContrast(imgGrayscale)

        height, width = imgGrayscale.shape

        imgBlurred = np.zeros((height, width, 1), np.uint8)
        
        imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, self.GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.ADAPTIVE_THRESH_BLOCK_SIZE, self.ADAPTIVE_THRESH_WEIGHT)

        return imgGrayscale, imgThresh

    def extractValue(self, imgOriginal):
        height, width, numChannels = imgOriginal.shape

        imgHSV = np.zeros((height, width, 3), np.uint8)

        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

        imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

        return imgValue

    def maximizeContrast(self, imgGrayscale):

        height, width = imgGrayscale.shape

        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        return imgGrayscalePlusTopHatMinusBlackHat
