import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from project.annotations import *
import re

class TrackPoint():
    
    def __init__(self, maxFramesToDisappear = 10):
        self.ID = 1
        self.points = dict()
        self.maxFramesToDisappear = maxFramesToDisappear

    def isValidNum(self, numplate):
        found = re.match("^[A-Za-z]{2,4}[0-9]{2,4}$", numplate)
        if (found == None):
            return False
        else:
            return True

    def updatePoints(self, points, maxDistance = 400):
        updatedPoints = []
        disappearedPoints = []

        if (len(self.points) == 0):
            for ID in points:

                pointCords = points[ID]["point"]
                numplate = points[ID]["numplate"]
                if (self.isValidNum(numplate)):
                    correctednumplate = numplate
                else:
                    correctednumplate = "Not Detected"

                self.registerPoint(pointCords, numplate, correctednumplate=correctednumplate)
        else:
            for ID in points:
                pointCords = points[ID]["point"]
                numplate = points[ID]["numplate"]
                
                ind, distance = self.findMinDistacne(self.points, pointCords)
                
                #if not valid number don't change previous number
                if (self.isValidNum(numplate)):
                    correctednumplate = numplate
                else:
                    correctednumplate = self.points[ind]["correctednumplate"]

                #print("Corrected Num Plate: ", correctednumplate)

                if (distance < maxDistance):
                    self.points[ind] = {
                        "point": pointCords,
                        "numplate": numplate,
                        "correctednumplate": correctednumplate,
                        "frame": 0
                    }
                    updatedPoints.append(ind)
                else: 
                    ind = self.registerPoint(pointCords, numplate, correctednumplate=correctednumplate)
                    updatedPoints.append(ind)
                
            #not updated points, possibly have gone off the scene
            for ID in self.points:
                if not (ID in updatedPoints):
                    self.points[ID]["frame"] += 1

                if self.points[ID]["frame"] > self.maxFramesToDisappear:
                    disappearedPoints.append(ID)
                
            self.removeDisappeardFromList(disappearedPoints)

            #print(self.points)

        return self.points
    
    def registerPointsFromAnnotations(self, annotations):
        plateAnnotations = annotations.getAnnotations()["plates"]
        points = dict()
        for count, plate in enumerate(plateAnnotations):
            points[count] = {"point" : (plate["left"], plate["top"]), "numplate": plate["numplate"]}

        self.updatePoints(points)

    def getNumPlateFromID(self, ID):
        return self.points[ID]["correctednumplate"]

    def getIDFromNumPlate(self, numplate):
        for plateID in self.points:
            if self.points[plateID]["numplate"] == numplate:
                return plateID
        return 0
    
    def getIDFromNumPlatePosition(self, left, top):
        for plateID in self.points:
            if self.points[plateID]["point"]["left"] == left and self.points[plateID]["point"]["top"] == top:
                return plateID, self.points[plateID]['numplate']
        return 0

    def removeDisappeardFromList(self, disappearedPoints):
        for pointID in disappearedPoints:
            self.removePoint(pointID)

    def registerPoint(self, pointCords, numplate, correctednumplate=''):
        if (correctednumplate == ''):
            correctednumplate = numplate

        self.points[self.ID] = {
            "point": pointCords,
            "numplate": numplate,
            "correctednumplate": correctednumplate,
            "frame": 0
        }
        self.ID += 1
        return (self.ID - 1)

    def removePoint(self, pointID):
        print("Car # ", pointID, " Plate Number: ", self.points[pointID]["correctednumplate"])
        self.points.pop(pointID)
        return True

    def findMinDistacne(self, points, point):
        distance = 100000000
        ind = -1
        for index in points:
            tempDist = dist.cdist([points[index]["point"]], [point]) [0][0]
            #print(tempDist)
            if (tempDist < distance):
                distance = tempDist
                ind = index

        return ind, distance

