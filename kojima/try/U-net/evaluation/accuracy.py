###python3 evaluation.py 1.png
import sys 
from PIL import Image
import numpy as np

def evalution(t_img , a_img):
    test = np.array(t_img) # 提案手法によって生成されたセグメン画像
    accuracy = np.array(a_img) # 正解のセグメン画像

    back_all = 0            #全背景数
    nuclear_all = 0         #全核数
    cytoplasm_all = 0       #全細胞質数
    accuracy_num = 0        #正解値と予測値が等しい数
    back_accuracy = 0       #背景の正解数
    nuclear_accuracy = 0    #核の正解数
    cytoplasm_accuracy = 0  #細胞質の正解数
    pixel_all = 0           #画像に含まれる画素数

    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            pixel_all += 1 # 1枚の画像に含まれる画素数を格納
            if accuracy[i][j] == 0:  #正解が背景なら
                back_all += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が背景なら
                    accuracy_num += 1
                    # back_accuracy += 1
            elif accuracy[i][j] == 1:  #正解が核なら
                nuclear_all += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が核なら
                    accuracy_num += 1
                    # nuclear_accuracy += 1
            elif accuracy[i][j] == 2:  #正解が細胞質なら
                cytoplasm_all += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が細胞質なら
                    accuracy_num += 1
                    # cytoplasm_accuracy += 1

    final_accuracy = accuracy_num / pixel_all # 正解数/データ数
    #print(args[1])
    #print("back:", back)
    #print("nuclear:", nuclear)
    #print("cytoplasm:", cytoplasm)
    print("正解率:"+str(final_accuracy))
    #print("")
#------------------main--------------------#
args = sys.argv

img = Image.open("Generated_seg_images/seg_" + args[1])
accuracy_img = Image.open("SegmentationClass/" + args[1])

evalution(img, accuracy_img)
