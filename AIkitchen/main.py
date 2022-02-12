import cv2
from cvzone_modified import HandDetector
import numpy as np
import cvzone  # cvzone == 1.4.1 #mediapipe == 0.8.7
from time import sleep
import time
from PIL import Image, ImageDraw, ImageFont
import tensorrt as trt
import engine as eng
import inference as inf
import os
#from cvzone.SelfiSegmentationModule import SelfiSegmentation
#gpu_frame = cv2.cuda_GpuMat
#segmentor = SelfiSegmentation()

# ==============================================camera setting =====================================================
img_length = 1280
img_width = 720
cap = cv2.VideoCapture(0)
cv2.namedWindow("AiKitchen", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "AiKitchen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap.set(3, img_length)
cap.set(4, img_width)

# ==================================================== setting ================================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

class_dict = {'rice': 0, 'meat': 1, 'fish': 2, 'tofu': 3, 'onion': 4,
              'greenbean': 5, 'egg': 6, 'carrot': 7, 'spinach': 8, 'ginger': 9}
#img_dir = './test_img'
onnx_path = './model/Best-model.onnx'
trt_path = './model/Best-model.trt'
num_classes = 10
IMG_SIZE = 224
input_shape = [1, IMG_SIZE, IMG_SIZE, 3]
BATCH_SIZE = 1
serialized_plan = './model/Best-model.plan'
trt_path = './model/Best-model.trt'

# =================================================================================================================
fpsReader = cvzone.FPS()

# ================================================== Other function setting   ======================================================================
def get_key(dict, value):
    key = [k for (k, v) in dict.items() if v == value]
    return key[0]


def get_predit_img(image, img_size):
    img = cv2.resize(image, (img_size, img_size))
    img = np.array(img, dtype=np.float32)
    img = img/127-1
    img = np.expand_dims(img, axis=0)
    return img


def drawAll(img):
    imgNew = np.zeros_like(img, np.uint8)
    cv2.rectangle(imgNew, (5, 5), (205, 205), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, "detect", (1,  100),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
    cv2.rectangle(imgNew, (5, 215), (205, 415), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, "next", (5,  300),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
    cv2.rectangle(imgNew, (5, 425), (205, 625), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, "finish", (5,  500),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
    cvzone.cornerRect(img, (5, 5, 200, 200),
                      20, rt=0)
    cvzone.cornerRect(img, (5, 215, 200, 200),
                      20, rt=0)
    cvzone.cornerRect(img, (5, 425, 200, 200),
                      20, rt=0)

    img[125:625, 210:533, :] = bag
    _, img = fpsReader.update(img, (210, 60))

    out = img.copy()
    alpha = 0.7
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out


# ============================================  bag image setting ========================================================
bag = cv2.resize(cv2.imread('ingredient/bag.png'), (323, 500))
# ============================================dish library setting=====================================================
dishes = []


class dish():
    def __init__(self, element, name):
        self.element = element
        self.name = name

    def getname(self):
        return self.name

    def getelement(self):
        return self.element


element1 = {"egg", "meat", "rice", "onion"}
#dish_img =cv2.imread('dish/0.jpg')
name1 = "三層肉炒飯"
dishtemp1 = dish(element1, name1)
dishes.append(dishtemp1)

element2 = {"meat", "spinach"}
#dish_img =cv2.imread('dish/1.jpg')
name2 = "沙茶豬肉菠菜"
dishtemp2 = dish(element2, name2)
dishes.append(dishtemp2)

element3 = {"fish", "tofu", "ginger"}
#dish_img =cv2.imread('dish/2.jpg')
name3 = "味噌豆腐魚片湯"
dishtemp3 = dish(element3, name3)
dishes.append(dishtemp3)

element4 = {"egg", "carrot", "onion"}
#dish_img =cv2.imread('dish/3.jpg')
name4 = "紅蘿蔔洋蔥炒蛋"
dishtemp4 = dish(element4, name4)
dishes.append(dishtemp4)

element5 = {"greenbean", "carrot"}
#dish_img =cv2.imread('dish/4.jpg')
name5 = "涼拌四季豆"
dishtemp5 = dish(element5, name5)
dishes.append(dishtemp5)

# ================================================ingredient setting =====================================================
ingredients = []


class ingredient():
    def __init__(self, img, name):
        self.img = img
        self.name = name

    def set(self, img, name):
        self.img = img
        self.name = name

    def getimg(self):
        return self.img

    def getname(self):
        return self.name


# ==================================================== Main Loop ===========================================================
# setting
ingredients_count = 0  # TEST
step3 = False
step2 = False
step1 = True
detect = False
next = True
detector = HandDetector(detectionCon=0.8, maxHands=1)
g = 0
h = 0

# little trick
# load a image from file before start
# because the speed predict the first image will be very slow(don't konw why...), so we let jetson nano to inference a test image before start.
img = cv2.imread('test_img.jpg')[:, :, ::-1]
img = get_predit_img(img, IMG_SIZE)
engine = eng.load_engine(trt_runtime, trt_path)   # TensorRT engine
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, BATCH_SIZE, trt.float32)
out = inf.do_inference(engine, img, h_input, d_input, h_output, d_output, stream, BATCH_SIZE, IMG_SIZE, IMG_SIZE)
print("start!!")


# step 1 --> step 2 --> step 3
# implement like State machine
# ==================================================        step 1   第一個頁面          =================================================================
while True:
    ingredient_pos = [505, 403]
    if step1:
        success, img = cap.read()
        img_for_hand1 = img[0:720, 0:640, :]  # 畫面左半邊給手
        img_for_ingredient = img[0:720, 640:1280, :]  # 畫面右半邊給食材

        hands = detector.findHands(img_for_hand1, draw=False)

        img = drawAll(img)

        if hands:
            hand1 = hands[0]
            lmList = hand1["lmList"]
            clear_distance = detector.findDistance(lmList[16], lmList[4], img_for_hand1)

			# "detect"
            if 5 < lmList[8][0] < 205 and 5 < lmList[8][1] < 205:   
                cv2.rectangle(img, (0, 0), (210, 210), (0, 255, 255), cv2.FILLED)
                cv2.putText(img, "detect", (1,  100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                d, _, _ = detector.findDistance(lmList[8], lmList[4], img_for_hand1)

                if d < 30 and next:  # 食指跟大拇指的距離小於 threshold:30
                    image = get_predit_img(img_for_ingredient[:, :, ::-1], IMG_SIZE)

                    # TensorRT engine
                    engine = eng.load_engine(trt_runtime, trt_path)   # faster
                    #engine = eng.load_engine(trt_runtime, serialized_plan)

                    # allocate buffers
                    h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, BATCH_SIZE, trt.float32)
                    start = time.time()
                    out = inf.do_inference(engine, image, h_input, d_input, h_output, d_output, stream, BATCH_SIZE, IMG_SIZE, IMG_SIZE)
                    end = time.time()

                    # print(out)
                    # example out : [2.    1.859 0.    0.    0.    0.    0.    0.    0.    0.   ]
                    index = np.argmax(out)
                    result = get_key(class_dict, index)
                    print("predict food is : ", result)
                    print("inference time : %.3f secs/frame" % (end - start))

                    if result:
                        detect = True
                        next = False
                        result_img = cv2.resize(cv2.imread('ingredient/'+result + '.jpg'), (100, 100))  # ex: egg.jpg
                        ingredients_count += 1

			# next
            elif 5 < lmList[8][0] < 205 and 215 < lmList[8][1] < 415:
                cv2.rectangle(img, (0, 210), (210, 420), (0, 255, 255), cv2.FILLED)
                cv2.putText(img, "next", (5,  300),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                d, _, _ = detector.findDistance(lmList[8], lmList[4], img_for_hand1)
                if d < 30 and detect:
                    detect = False
                    next = True
                    if ingredients_count == 1:  # 應該可以換種寫法
                        temp1 = ingredient(result_img, result)  # ingredient(img, name)
                        ingredients.append(temp1)
                    if ingredients_count == 2:
                        temp2 = ingredient(result_img, result)
                        ingredients.append(temp2)
                    if ingredients_count == 3:
                        temp3 = ingredient(result_img, result)
                        ingredients.append(temp3)
                    if ingredients_count == 4:
                        temp4 = ingredient(result_img, result)
                        ingredients.append(temp4)
                    if ingredients_count == 5:
                        temp5 = ingredient(result_img, result)
                        ingredients.append(temp5)
                    if ingredients_count == 6:
                        temp6 = ingredient(result_img, result)
                        ingredients.append(temp6)
                    if ingredients_count == 7:
                        temp7 = ingredient(result_img, result)
                        ingredients.append(temp7)
                    if ingredients_count == 8:
                        temp8 = ingredient(result_img, result)
                        ingredients.append(temp8)
                    if ingredients_count == 9:
                        temp9 = ingredient(result_img, result)
                        ingredients.append(temp9)
                    if ingredients_count == 10:
                        temp10 = ingredient(result_img, result)
                        ingredients.append(temp10)

			# finish
            elif 5 < lmList[8][0] < 205 and 425 < lmList[8][1] < 625:
                cv2.rectangle(img, (0, 420), (210, 630),
                              (0, 255, 255), cv2.FILLED)
                cv2.putText(img, "finish", (5,  500),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                d, _, _ = detector.findDistance(
                    lmList[8], lmList[4], img_for_hand1)
                if d < 30:
                    for y in range(len(dishes)):
                        i = 0
                        for x in ingredients:
                            if x.getname() in dishes[y].getelement():
                                # print(x.getname())
                                i += 1
                        if i == len(dishes[y].getelement()) and i > 0:
                            h = y
                            step1 = False
                            step2 = True
                            sleep(1)
                            break
                        else:
                            img[318:402, 486:794, :] = cv2.imread(
                                "dish/error.jpg")

    if detect:  # 將detect的結果(文字及照片)，顯示在畫面中間                                                                      #TEST
        if clear_distance < 30:
            detect = False
            next = True
        cv2.putText(img, result, (590, 360),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  # TEST
        img[215:315, 590:690, :] = result_img

     # 應該可以換種寫法，不用每加一個ingredient就重新更新，ingredient bag的圖片
    for x in ingredients:  # 將辨識完的食材放到ingredient bag的圖裡
        img[ingredient_pos[0]:ingredient_pos[0]+100,
            ingredient_pos[1]:ingredient_pos[1]+100, :] = x.getimg()
        if ingredient_pos[0] == 305:
            ingredient_pos[0] = 505
            ingredient_pos[1] -= 100
        else:
            ingredient_pos[0] -= 100  # 新加的食材會再舊的食材上方


# =============================================          step 2    第二個頁面      ==============================================================
    ingredient_pos = [505, 403]

    if step2:
        g += 1
        
        for dish in range(len(dishes)) :
            n=0
            for ingredient in ingredients :
                if ingredient.getname() in dishes[dish].getelement():  # match ingredients with dishes
                    n+=1
            if len(ingredients) > 0:
                h=dish
                img_step2_bg = cv2.resize(cv2.imread('dish/'+str(dish)+'_bg.jpg'),(1280,720))
                img = img_step2_bg
                img[248:580,737:1180,:] = cv2.imread('dish/'+str(dish)+'.jpg')
                break

        for ingredient in ingredients:  # ingredients bag
            img[ingredient_pos[0]:ingredient_pos[0]+100,
                ingredient_pos[1]-50:ingredient_pos[1]+50, :] = ingredient.getimg()
            if ingredient_pos[0] == 305:
                ingredient_pos[0] = 505
                ingredient_pos[1] -= 100
            else:
                ingredient_pos[0] -= 100

    if g > 10:
        sleep(10)
        step2 = False
        step3 = True
        g = 0

# =============================================          step 3  第三個頁面         =============================================================
    if step3:
        g += 1
        #cv2.rectangle(img, (0, 0), (1280 , 720), (255, 255, 255),cv2.FILLED )
        img = cv2.resize(cv2.imread('recipe/'+str(h)+'.jpg'), (1280, 720))
        if g > 10:
            sleep(120)
            cv2.destroyWindow("AiKitchen")
            break

        # break
    cv2.imshow("AiKitchen", img)
    cv2.waitKey(1)
del(cap)
