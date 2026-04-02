import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import sys
import mediapipe as mp

#------------------------------------------------------------------
# クラスラベル定義(自分で想定した手形状のクラスを定義すること)
#class_names = ['rock', 'paper', 'scissors']
#
class_names = ['one','two','three','four','five']
#------------------------------------------------------------------


# データパス
data_path = './data/'

# 特徴点データとクラスラベルを返す
# train_features, train_labels：訓練用データ
# test_features, test_labels：テスト用データ
def load_data():

    path = data_path
    files = glob.glob(os.path.join(path, '*/*.csv'))
  
    features = np.ndarray((len(files), 21, 2), dtype=np.int64) 
    labels = np.ndarray(len(files), dtype=np.uint8)

    for idx, file in enumerate(files):
       df = pd.read_csv(file, index_col=0)
       features[idx] = df.values
       label = os.path.split(os.path.dirname(file))[-1]
       #print(label)
       labels[idx] = class_names.index(label)
       
    train_features, test_features, train_labels, test_labels \
        = train_test_split(features, labels, test_size=0.3, random_state=0, stratify=labels)
    
    return train_features, test_features, train_labels, test_labels


# 手形状を撮影しデータを保存
def hand_capture(capture_num = 50):
    
    foldername = data_path
    if(os.path.exists(foldername)==False):
        os.mkdir(foldername)

    if(os.path.exists(foldername)):
        for idx, cn in enumerate(class_names):
            print(idx, ':', cn)

    print('カテゴリ番号を入力して下さい')   
    class_id = input('>> ')
    if(int(class_id) >= len(class_names)):
        sys.exit()
    print('カテゴリ {} の手形状データを取得します'.format(class_names[int(class_id)]) )
    
    foldername += class_names[int(class_id)]
    fname = class_names[int(class_id)]
    #print(fname)
    if(os.path.exists(foldername)==False):
        os.mkdir(foldername)
    
    print('データ保存したいところで [s]キーを入力して下さい')

    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit()

    # Hand Landmark Model
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode = False, 
                          max_num_hands = 1, # 検出する手の最大数
                          min_detection_confidence = 0.5, 
                          min_tracking_confidence = 0.5) 
    mpDraw = mp.solutions.drawing_utils
    
    cap_count = 0
    while cap_count < capture_num:
    
        ret, img = cap.read()
    
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
          
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                d_lm = [] 
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    d_lm.append([cx, cy]) # 特徴量を変更したい場合はここを変更
 
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
                if key & 0xFF == ord('s'):
                    fname1 = fname + '_' + ('%.3d' % cap_count)
                    filename = foldername + '/' + fname1 + '.csv'
                    print('データ取得... {0} 個目: ({1})'.format(cap_count+1,filename))
                    
                    # DataFrameを使って出力
                    df = pd.DataFrame(d_lm, columns=['cx', 'cy'])
                    df.to_csv(filename) # すべての部位を保存

                    cap_count += 1
    
         
        fimg = cv2.flip(img, 1) # 左右反転
        cv2.putText(fimg, str(cap_count), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("capture　frame", fimg)
       
    
    cap.release() # ビデオキャプチャのメモリ解放
    cv2.destroyAllWindows() # すべてのウィンドウを閉じる
