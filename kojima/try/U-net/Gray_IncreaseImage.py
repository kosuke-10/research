######################################################
# python3 Gray_IncreaseImage.py 20
######################################################

import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageOps
import sys


# Histogram homogenization function
def equalizeHistRGB(src):
    
    RGB = cv2.split(src)
    Blue   = RGB[0]
    Green = RGB[1]
    Red    = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])
    img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
    return img_hist

# Gaussian noise function
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss
    return noisy

# Salt & Pepper noise function
def addSaltPepperNoise(src):
    row,col,ch = src.shape
    s_vs_p = 0.5
    amount = 0.004
    out = src.copy()

    # Salt mode
    try:
        num_salt = np.ceil(amount * src.size * s_vs_p)
        coords = [np.random.randint(0, i-1 , int(num_salt)) for i in src.shape]
        out[coords[:-1]] = (255,255,255)
    except:
        pass

    # Pepper mode
    try:
        num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in src.shape]
        out[coords[:-1]] = (0,0,0)
    except:
        pass
    return out

# Rotation
def rotate_image(src1, src2, angle):
    orig_h, orig_w = src1.shape[:2]
    matrix = cv2.getRotationMatrix2D((orig_w/2, orig_h/2), angle, 1)
    return cv2.warpAffine(src1, matrix, (orig_w, orig_h), src1, flags=cv2.INTER_LINEAR), src2.rotate(angle)

def gray(img, count):
    value = 0
    probability = np.random.randint(2)
    if probability == 0:  # グレイスケール値を上げる場合
        if count == 0:
            value = 255 - np.amax(img) 
            count += 1
        elif 255 - np.amax(img) >= 50:
            value = np.random.randint(50)
        elif 255 - np.amax(img) >= 40:
            value = np.random.randint(40)
        elif 255 - np.amax(img) >= 30:
            value = np.random.randint(30)
        elif 255 - np.amax(img) >= 20:
            value = np.random.randint(20)
        elif 255 - np.amax(img) >= 10:
            value = np.random.randint(10)  # グレイスケール値の変更幅を取得
        else:
            value = 0
        #print("max:",(255 - np.amax(img)))
        #print(value)
        img = img + value
    elif probability == 1:  # グレイスケールを下げる
        #print("min:", np.amin(img))
        if count == 1:
            value = np.amin(img) 
            count += 1
        elif np.amin(img) >= 20:
            value = np.random.randint(20)
        elif np.amin(img) >= 15:
            value = np.random.randint(15)
        elif np.amin(img) >= 10:
            value = np.random.randint(10)
        elif np.amin(img) >= 5:
            value = np.random.randint(5)
        else:
            value = 0
        #print(value)
        img = img - value
    return img, count

def contrast(img):
    value = np.random.uniform(0.6, 1.2)
    #print("value:",value)
    img = img * value
    return np.clip(img, 0, 255).astype(np.uint8)

def gamma_correction(img):
    gamma = np.random.uniform(0.6, 1.2)
    table = (np.arange(256)/255) ** gamma * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


args = sys.argv

######################################################
increase_num = args[1]
######################################################
dataset = "dataset"

img_filesJ = sorted(glob.glob("../mnt/" + dataset + "/JPEGImages/*"))  # jpg画像の読み込み
img_filesS = sorted(glob.glob("../mnt/" + dataset + "/SegmentationClass/*"))  # seg画像の読み込み
JPEG_out_base_path = "../mnt/" + dataset + "/JPEGImagesOUT"  # jpg出力先のパスを指定
SEGM_out_base_path = "../mnt/" + dataset + "/SegmentationClassOUT"  # seg出力先のパスを指定


imgs = []
for (img_fileJ, img_fileS) in zip(img_filesJ, img_filesS):
    imgs.append([cv2.imread(img_fileJ, cv2.IMREAD_UNCHANGED), Image.open(img_fileS)])

# Generate lookup table
min_table = 50
max_table = 205
diff_table = max_table - min_table
gamma1 = 0.75
gamma2 = 1.5
LUT_HC = np.arange(256, dtype = 'uint8')
LUT_LC = np.arange(256, dtype = 'uint8')
LUT_G1 = np.arange(256, dtype = 'uint8')
LUT_G2 = np.arange(256, dtype = 'uint8')
LUTs = []
# Smoothing sequence
average_square = (10,10)
# Create high contrast LUT
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table                        
for i in range(max_table, 255):
    LUT_HC[i] = 255
# Other LUT creation
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1) 
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
LUTs.append(LUT_HC)
LUTs.append(LUT_LC)
LUTs.append(LUT_G1)
LUTs.append(LUT_G2)

imgcnt = 0

for img in imgs:
    count = 0
    for i in range(int(increase_num)):
        jpgimg = img[0]
        segimg = img[1]

        # Gamma correction　ガンマ補正
        if np.random.randint(2) == 1:
            jpgimg = gamma_correction(jpgimg)

        # グレイスケールの濃淡を変化
        if np.random.randint(2) == 1:
            jpgimg,count = gray(jpgimg, count)
            
        # Contrast conversion execution
        #if np.random.randint(2) == 1:
         #   level = np.random.randint(4)
          #  jpgimg = cv2.LUT(jpgimg, LUTs[level])

        #Smoothing execution 平滑化
        # if np.random.randint(2) == 1:
            # jpgimg = cv2.blur(jpgimg, average_square)

        # Histogram equalization execution
        #if np.random.randint(2) == 1:
        #    jpgimg = equalizeHistRGB(jpgimg)

        # Gaussian noise addition execution  ガウシアンノイズ追加
        # if np.random.randint(2) == 1:
            # jpgimg = addGaussianNoise(jpgimg)

        # Salt & Pepper noise addition execution  ソルト＆ペッパーノイズ追加
        if np.random.randint(2) == 1:
            jpgimg = addSaltPepperNoise(jpgimg)

        #Rotation  回転
        #if np.random.randint(2) == 1:
        #    angle = 360/int(increase_num)*i
        #    jpgimg, segimg = rotate_image(jpgimg, segimg, angle)

        # Reverse execution　左右反転
        if np.random.randint(2) == 1:
            jpgimg = cv2.flip(jpgimg, 1)
            segimg = ImageOps.mirror(segimg)

        # Reverse execution  上下反転
        if np.random.randint(2) == 1:
            jpgimg = cv2.flip(jpgimg, 0)
            segimg = ImageOps.flip(segimg)

        # Contrast　コントラスト調整
        if np.random.randint(2) == 1:
            jpgimg = contrast(jpgimg)


        # Image storage after padding
        JPEG_image_path = "%s/%04d_%04d.jpg" % (JPEG_out_base_path, imgcnt, i)
        SEGM_image_path = "%s/%04d_%04d.png" % (SEGM_out_base_path, imgcnt, i)
        cv2.imwrite(JPEG_image_path, jpgimg)
        segimg.save(SEGM_image_path)

        print("imgcnt =", imgcnt, "num =", i)
    imgcnt += 1

print("Finish!!")
