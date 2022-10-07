import numpy as np
from datetime import datetime, timedelta, timezone
import cv2
import imutils
import os
from imutils import contours
import json

from tensorflow.keras.models import load_model

class Ticket():
    def __init__(self, image, type) -> None:
        self.image = image
        self.type = type
        self.nums = []
        self.date = ""
        self.model = load_model('my_model_chars_74')
        self.model_mnist = load_model('my_model')
        self.model_alphabet = load_model('my_model_alphabet')
        self.mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.account = json.load(open("./account.json", "r")) if os.path.exists("./account.json") else {}
        self.rois = {}
        if self.account != {}:
            for key in list(self.account[self.type]["number_region"].keys()):
                self.rois[key] = {"img":[], "info":[], "predictions":[], "confident":[], "win":[]}

    def find_rois(self, test=False) -> bool:
        if self.account == {}: 
            print("No account.json file")
            return False
        image = cv2.imread(self.image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

        cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        height, width = threshed.shape
        # Trial and Error Debug section of code
        if test:
            for c in cnts:
                # compute the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)
                x_val = x / width
                # if the contour is sufficiently large, it must be a digit 
                if (w >= 11 and w <=20) and (h >= 10 and h <= 50) and (0.73 <= x_val <= 0.75):
                    print(x, y, w, h)
                    print(x_val)
                    cv2.drawContours(image, [c], 0, (0,255,0), 3)
                    cv2.imshow('thresh', image)
                    cv2.waitKey()
            return
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit and (x >=  and x <= )
            x_val = x / width
            for key in list(self.rois.keys()):
                checks = self.account[self.type]["number_region"][key]
                if (checks["w"][0] <= w <= checks["w"][1]) and (checks["h"][0] <= h <= checks["h"][1]) and (checks["x"][0] <= x_val <= checks["x"][1]):
                    if key == "MegaBall" and width > 100:
                        self.rois["MegaBall"]["img"].append(thresh[y:y + h, x:x + int(w/2)])
                        self.rois["MegaBall"]["info"].append(c)
                    else:
                        self.rois[key]["img"].append(thresh[y:y + h, x:x + w])
                        self.rois[key]["info"].append(c)
        return True

    def predict(self, model, image):
        im1 = cv2.resize(image, dsize=(28, 28))
        im1_array = np.array(im1) / 255.0
        return model.predict(im1_array.reshape(1, 28, 28, 1))
    
    def extract_info(self) -> bool:   
        print("Making predictions")     
        for key in list(self.rois.keys()):
            if key == "DateOfDrawing":
                for img in self.rois[key]["img"]:
                    # cv2.imshow('img', img)

                    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cnts = imutils.grab_contours(cnts)
                    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
                    result = []
                    conf = 1
                    for c in cnts:
                        (x, y, w, h) = cv2.boundingRect(c)
                        if w > 10 and h > 15:
                            temp = img[y:y + h, x:x + w]
                            if len(result) < 6:
                                pred = self.predict(self.model_alphabet, temp)
                                conf *= pred.max()
                                result.append(self.mapping[pred.argmax()])
                            else:
                                pred = self.predict(self.model, temp)
                                conf *= pred.max()
                                result.append(f"{pred.argmax()}")
                    date = datetime.strptime("".join(result), '%a%b%d%Y')
                    self.rois[key]["predictions"].append(date)
                    self.rois[key]["confident"].append(conf > 0.6)
                self.date =  self.rois[key]["predictions"][0]
            else:
                for img in self.rois[key]["img"]:
                    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
                        
                    # # kernel = np.ones((4,4),np.uint8)
                    # kernel = np.ones((2,2),np.uint8)
                    # # dilation = cv2.dilate(img,kernel,iterations = 1)
                    # # cv2.imshow('dialation', dilation)
                    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                    #cv2.imshow('closing', img)


                    #cv2.imshow('roi', img)
                    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cnts = imutils.grab_contours(cnts)
                    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
                    result = []
                    conf = 1
                    for c in cnts:
                        (x, y, w, h) = cv2.boundingRect(c)
                        if w > 10 and h > 15:
                            temp = img[y:y + h, x:x + w]
                            pred = self.predict(self.model, temp)
                            conf *= pred.max()
                            result.append(f"{'' if pred.argmax() == 0 and len(result) == 0 else pred.argmax()}")
                    # print("".join(result))
                    # if not conf > 0.7:
                    #     print(conf)
                    #     print("".join(result))
                    #     cv2.imshow('img', img)
                    #     cv2.waitKey()
                    self.rois[key]["predictions"].append("".join(result))
                    self.rois[key]["confident"].append(conf > 0.6)
        return True

    def generate_nums(self) -> bool:
        if len(self.rois["col1"]["predictions"]) == len(self.rois["col2"]["predictions"]) == len(self.rois["col3"]["predictions"]) == len(self.rois["col4"]["predictions"]) == len(self.rois["col5"]["predictions"]) == len(self.rois["MegaBall"]["predictions"]):
            #all tickets were found
            for i in range(len(self.rois["col1"]["predictions"])-1, -1, -1):
                row = list([self.rois[f"col{x}"]["predictions"][i] for x in range(1,6)])
                row.append(self.rois["MegaBall"]["predictions"][i])
                self.nums.append(row)
        else:
            print("Rows of numbers not generated")
        return True
    
    def process_ticket(self) -> bool:
        if not self.find_rois(): return "Something went wrong when finding rois"
        if not self.extract_info(): return "Something went wrong when extracting info"
        if not self.generate_nums(): return "Something went wrong when generating nums"
        return True
    
    def save_results(self) -> None:
        if len(self.rois["col1"]["win"]) == 0: 
            print("Numbers not checked yet")
            return
        image = cv2.imread(self.image)
        for key in list(self.rois.keys()): 
            if key != "DateOfDrawing":
                for i in range(0,len(self.rois[key]["win"])):
                    if not self.rois[key]["confident"][i]:
                        c = self.rois[key]["info"][i]
                        cv2.drawContours(image, [c], 0, (0,255,255), 5, lineType=cv2.LINE_8)
                    if self.rois[key]["win"][i]:
                        c = self.rois[key]["info"][i]
                        cv2.drawContours(image, [c], 0, (0,255,0), 3)
            else:
                for i in range(0,len(self.rois[key]["predictions"])):
                    if not self.rois[key]["confident"][i]:
                        c = self.rois[key]["info"][i]
                        cv2.drawContours(image, [c], 0, (0,255,255), 5, lineType=cv2.LINE_8)
        # cv2.imshow('results', image)
        # cv2.waitKey()
        fn = self.image.replace("_output.png", "_results.png")
        cv2.imwrite(f"{fn}", image)
        return

if __name__ == "__main__":
    ticket = Ticket("./IMG_2342_output.png", "MegaMillions")
    ticket.find_rois(test=True)