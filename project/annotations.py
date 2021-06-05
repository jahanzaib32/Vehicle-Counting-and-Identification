import cv2
import os

class SceneAnnotations():
    def __init__(self, img):
        self.image = img
        self.annotations = dict()
        self.annotations["cars"] = []
        self.annotations["plates"] = []

    def getImage(self):
        return self.image

    def getAnnotations(self):
        return self.annotations

    def addCar(self, left, top, right, bottom, model="N/A", color="N/A"):
        self.annotations["cars"].append({
            "model": model,
            "color": color,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        })
    
    def addPlate(self, left, top, right, bottom, numplate="N/A"):
        self.annotations["plates"].append({
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "numplate": numplate
        })