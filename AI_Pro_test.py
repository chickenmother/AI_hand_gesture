import tensorflow as tf
from tensorflow import keras
import cv2
import mediapipe as mp
import numpy as np
import handsign
import random
import math
import time
import pygame
import pygame.mixer

pygame.mixer.init()

se1=pygame.mixer.Sound("SE/SFX_ Point - Flappy Bird.mp3")
pygame.mixer_music.load("SE/Entrance (Deemo Version).mp3")
pygame.mixer.music.play(-1)



#define
gameWave=0
startTime=time.time()
prevTime=time.time()
cal=False
points=0

add_target=random.randint(1,100)
add_ans=random.randint(1,5)
add_sum=add_target-add_ans

# Hand Landmark Model（手の領域内の21個のランドマークを検出するモデル）
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode = False, 
                      max_num_hands = 1,
                      min_detection_confidence = 0.5, 
                      min_tracking_confidence = 0.5) 
mpDraw = mp.solutions.drawing_utils

# クラスラベル定義（handsign.pyにて定義済み）
class_names = handsign.class_names

# モデルの読み込み
filename = './handsign_NN.h5'
model = keras.models.load_model(filename)

cap = cv2.VideoCapture(0) # ビデオキャプチャ準備

class enemy:
    def __init__(self,health, EneType):
        self.health=health
        self.EneType=EneType
    def demage(self):
        self.health-=1

enemy1=enemy(3,0)
enemy2=enemy(6,1)
enemy3=enemy(8,2)
boss=enemy(15,3)


def addition(pic,pred,sum,target,ans,elapsedTime):
    if elapsedTime<=25:
        question=str(sum)+" + ?="+str(target)
        cv2.putText(pic,question,(300,100),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)
        if ans-1==pred:
            cv2.putText(pic,"correct!!",(150,150),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)
            return True
    elif gameWave<3:
        results="gameover!! you got "+str(points)+ " points!"
        cv2.putText(pic,results,(50,250),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255), 3)
        pygame.mixer.music.stop()
    elif gameWave>=3:
        results="Congrats! You've cleared all the Stages!"
        results2=" you got "+str(points)+ " points!"
        cv2.putText(pic,results,(10,250),cv2.FONT_HERSHEY_PLAIN,1.85,(255,0,255), 3)
        cv2.putText(pic,results2,(150,280),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255), 3)
        pygame.mixer.music.stop()

while True:
    currentTime=time.time()
    elapsedTime=currentTime-startTime
    timeDiff=currentTime-prevTime
    prevTime=currentTime
    elapsedTime=round(elapsedTime)
    print("elapsed time: ",elapsedTime,"time per one frame:",timeDiff)
    success, img = cap.read()
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    xmin = ymin = xmax = ymax = 0
    pred_class = -1
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            d_lm = [] 
            d_lmx = []
            d_lmy = []            
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                d_lm.append([cx, cy])
                d_lmx.append(cx)
                d_lmy.append(cy)                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            feat = np.array(d_lm).reshape(1,21,2)
            pred = model.predict(feat)
            pred_class = np.argmax(pred)
            xmax = max(d_lmx)
            xmin = min(d_lmx)
            ymax = max(d_lmy)
            ymin = min(d_lmy)
    fimg = cv2.flip(img, 1) # 左右反転
    xmin, xmax  = w - xmax, w - xmin #座標も左右反転
    hx = (xmax+xmin)/2
    hy = (ymax+ymin)/2
    print(hx,hy)
    # xmin,ymin 右上、xmax,ymax 左下
    cv2.rectangle(fimg, (xmin, ymin),(xmax, ymax),(255,0,0),thickness=1)
    # hx,hy 手の中心
    cv2.ellipse(fimg,((hx, hy),(50,50),0),(0,0,255),thickness=1) 
    
    

    cal=addition(fimg,pred_class,add_sum,add_target,add_ans,elapsedTime)
    if cal==True:
        add_target=random.randint(1,100)
        add_ans=random.randint(1,5)
        add_sum=add_target-add_ans
        points+=1
        se1.play()
        if gameWave==0:
            enemy1.demage()
            if enemy1.health==0:
                gameWave+=1
        elif gameWave==1:
            enemy2.demage()
            if enemy2.health==0:
                gameWave+=1
        elif gameWave==2:
            enemy3.demage()
            if enemy3.health==0:
                gameWave+=1  


    addition(fimg,pred_class,add_sum,add_target,add_ans,elapsedTime)
    
    cv2.putText(fimg,"Wave: "+str(gameWave),(0,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)
    cv2.putText(fimg,str(points),(500,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)
    disptime="Time: "+str(elapsedTime)
    cv2.putText(fimg,disptime,(250,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)

    if pred_class != -1:
        cv2.putText(fimg, class_names[pred_class], (10,100), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", fimg)
    key = cv2.waitKey(1)
    if key != -1:
        break
    if elapsedTime==30:
        break

cap.release() # ビデオキャプチャのメモリ解放
cv2.destroyAllWindows() # すべてのウィンドウを閉じる