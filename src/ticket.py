import numpy as np
import cv2
import imutils

from tensorflow.keras.models import load_model

class Ticket():
    def __init__(self, image, type) -> None:
        self.image = image
        self.type = type
        self.nums = []
        self.date = ""
        self.model = load_model('my_model_chars_74')
        self.model_alphabet = load_model('my_model_alphabet')
        self.mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.rois = {}
        for i in range(1,6):
            self.rois[f"col{i}"] = {"img":[], "info":[], "predictions":[], "win":[]}
        self.rois["MegaBall"] = {"img":[], "info":[], "predictions":[], "win":[]}
        self.rois["DateOfDrawing"] = {"img":[], "info":[], "predictions":[]}

    def find_rois(self) -> bool:
        image = cv2.imread(self.image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

        cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # Trial and Error Debug section of code
        # loop over the digit area candidates
        # for c in cnts:
        #     # compute the bounding box of the contour
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     # if the contour is sufficiently large, it must be a digit 
        #     if (w >= 11 and w <=160) and (h >= 10 and h <= 50) and (x >= 620 and x <= 640):
        #         print(x, y, w, h)
        #         cv2.drawContours(image, [c], 0, (0,255,0), 3)
        #         cv2.imshow('thresh', image)
        #         cv2.waitKey()
        # return

        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit and (x >=  and x <= )
            if (w >= 11 and w <=160) and (h >= 10 and h <= 50): 
                if (x >= 155 and x <= 165):
                    self.rois[f"col{1}"]["img"].append(thresh[y:y + h, x:x + w])
                    self.rois[f"col{1}"]["info"].append(c)
                elif (x >= 275 and x <= 290):
                    self.rois[f"col{2}"]["img"].append(thresh[y:y + h, x:x + w])
                    self.rois[f"col{2}"]["info"].append(c)
                elif (x >= 390 and x <= 420):
                    self.rois[f"col{3}"]["img"].append(thresh[y:y + h, x:x + w])
                    self.rois[f"col{3}"]["info"].append(c)
                elif (x >= 510 and x <= 530):
                    self.rois[f"col{4}"]["img"].append(thresh[y:y + h, x:x + w])
                    self.rois[f"col{4}"]["info"].append(c)
                elif (x >= 620 and x <= 640):
                    self.rois[f"col{5}"]["img"].append(thresh[y:y + h, x:x + w])
                    self.rois[f"col{5}"]["info"].append(c)
                elif (x >= 855 and x <= 865):
                    self.rois["MegaBall"]["img"].append(thresh[y:y + h, x:x + int(w/2)])
                    self.rois["MegaBall"]["info"].append(c)
            elif (w >= 160 and w <=500) and (h >= 10 and h <= 50) and (x >= 610 and x <= 630):
                self.rois["DateOfDrawing"]["img"].append(thresh[y:y + h, x:x + w])
                self.rois["DateOfDrawing"]["info"].append(c)
        return True
    
    def extract_info(self) -> bool:        
        for key in list(self.rois.keys()):
            if key == "DateOfDrawing":
                for img in self.rois[key]["img"]:
                    # cv2.imshow('roi', img)
                    height, width = img.shape
                    results = []
                    #Gets the first 3 letters
                    for i in range(0,3):
                        imi = img[0:height, int(i * width / 16):int((i+1)*width/16)]
                        imi = cv2.resize(imi, dsize=(28, 28))
                        imi_array = np.array(imi) / 255.0
                        pred1 = self.model_alphabet.predict(imi_array.reshape(1, 28, 28, 1))
                        results.append(self.mapping[pred1.argmax()])
                    # for i in range(0,3):
                    #     imi = img[0:height, int((width / 5)+ i * width / 15):int((width / 5)+ i+1 * width / 15)]
                    #     cv2.imshow('imi', imi)
                    #     cv2.waitKey()
                    #     return
                    #     imi = cv2.resize(imi, dsize=(28, 28))
                    #     imi_array = np.array(imi) / 255.0
                    #     pred1 = model_alphabet.predict(imi_array.reshape(1, 28, 28, 1))
                    #     results.append(mapping[pred1.argmax()])
                    # print(results)
                    # print(nums)
                    # cv2.imshow('roi', img)
                    # cv2.waitKey()

            else:
                for im in self.rois[key]["img"]:
                    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
                        
                    # kernel = np.ones((4,4),np.uint8)
                    kernel = np.ones((2,2),np.uint8)
                    dilation = cv2.dilate(im,kernel,iterations = 1)
                    # cv2.imshow('dialation', dilation)
                    img = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
                    #cv2.imshow('closing', closing)
                    # kernel = np.ones((2,2),np.uint8)
                    # erosion = cv2.erode(closing,kernel,iterations = 1)
                    #cv2.imshow('erosion after dialation and closing', erosion)
                    #kernel = np.ones((6,6),np.uint8)
                    #closing2 = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
                    #cv2.imshow('closing2', closing2)

                    # if key == "MegaBall":
                    #     cv2.imshow('before', im)
                    #     cv2.imshow('after', img)
                    #     cv2.waitKey()

                    #cv2.imshow('roi', img)
                    height, width = img.shape
                    im1 = img[0:height, 0:int(width/2)]
                    im2 = img[0:height, int(width/2):width]
                    im1 = cv2.resize(im1, dsize=(28, 28))
                    im1_array = np.array(im1) / 255.0
                    im2 = cv2.resize(im2, dsize=(28, 28))
                    im2_array = np.array(im2) / 255.0
                    #cv2.imshow('img', img)
                    pred1 = self.model.predict(im1_array.reshape(1, 28, 28, 1))
                    pred2 = self.model.predict(im2_array.reshape(1, 28, 28, 1))
                    #print(f"{'' if pred1.argmax() == 0 else pred1.argmax()}{pred2.argmax()}")
                    #cv2.waitKey()
                    self.rois[key]["predictions"].append(f"{'' if pred1.argmax() == 0 else pred1.argmax()}{pred2.argmax()}")
        return True

    def generate_nums(self) -> bool:
        if len(self.rois["col1"]["predictions"]) == len(self.rois["col2"]["predictions"]) == len(self.rois["col3"]["predictions"]) == len(self.rois["col4"]["predictions"]) == len(self.rois["col5"]["predictions"]) == len(self.rois["MegaBall"]["predictions"]):
            #all tickets were found
            for i in range(len(self.rois["col1"]["predictions"])-1, -1, -1):
                row = list([self.rois[f"col{x}"]["predictions"][i] for x in range(1,6)])
                row.append(self.rois["MegaBall"]["predictions"][i])
                self.nums.append(row)
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
                    if self.rois[key]["win"][i]:
                        c = self.rois[key]["info"][i]
                        cv2.drawContours(image, [c], 0, (0,255,0), 3)
        # cv2.imshow('results', image)
        # cv2.waitKey()
        fn = self.image.replace("_output.png", "_results.png")
        cv2.imwrite(f"{fn}", image)
        return