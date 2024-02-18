import os


import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
classIDBin = 0
# Import all the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the waste images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# 0 = Recyclable
# 1 = Hazardous
# 2 = Food
# 3 = Residual

classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2
            }

#webcam = True
#cap = cv2.VideoCapture(0)

x=550
y=360

# Read class names from labels.txt
with open('Resources/Model/labels.txt', 'r') as file:
    class_names = file.read().splitlines()

# Mapping of class names to their respective indices
class_names_mapping = {class_names[i]: i for i in range(len(class_names))}


while True:
    #if webcam:success,img = cap.read()
    #else: img = cv2.imread(path)


    _, img = cap.read()
    imgResize = cv2.resize(img, (x, y))

    imgBackground = cv2.imread('Resources/WASTE SEGREGATOR.png')

    predection = classifier.getPrediction(img)

    classID = predection[1]
    print(classID)
    if classID != 0:
        # imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[classID]

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    imgBackground[148:148 + y, 159:159 + x] = imgResize

    predicted_class_name = class_names[classID]  # Get the predicted class name
    cv2.putText(imgBackground, predicted_class_name, (950, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Displays
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)