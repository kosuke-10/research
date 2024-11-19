###python3 evaluation.py 1.png
import sys 
from PIL import Image
import numpy as np

def evalution(t_img , a_img):
    test = np.array(t_img) # 提案手法によって生成されたセグメン画像
    accuracy = np.array(a_img) # 正解のセグメン画像

    back_all = 0            #全予測背景数
    nuclear_all = 0         #全予測核数
    cytoplasm_all = 0       #全予測細胞質数
    back_accuracy = 0       #予測背景数の中の正解数
    nuclear_accuracy = 0    #予測核数の中のの正解数
    cytoplasm_accuracy = 0  #予測細胞質数の中の正解数

    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if test[i][j] == 0:  #予測が背景なら
                back_all += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が背景なら
                    back_accuracy += 1
            elif test[i][j] == 1:  #予測が核なら
                nuclear_all += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が核なら
                    nuclear_accuracy += 1
            elif test[i][j] == 2:  #予測が細胞質なら
                cytoplasm_all += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が細胞質なら
                    cytoplasm_accuracy += 1
                    
    back = back_accuracy/back_all                   #背景の適合率
    nuclear = nuclear_accuracy/nuclear_all          #核の適合率
    cytoplasm = cytoplasm_accuracy/cytoplasm_all    #細胞質の適合率

    final_precision = (back + nuclear + cytoplasm)/3 #適合率のマクロ平均
    #print(args[1])
    #print("back:", back)
    #print("nuclear:", nuclear)
    #print("cytoplasm:", cytoplasm)
    print("適合率:"+str(final_precision))
    #print("")
#------------------main--------------------#
args = sys.argv

img = Image.open("Generated_seg_images/seg_" + args[1])
accuracy_img = Image.open("SegmentationClass/" + args[1])

evalution(img, accuracy_img)
