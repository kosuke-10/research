###python3 evaluation.py 1.png
import sys 
from PIL import Image
import numpy as np

def evalution(t_img , a_img):
    test = np.array(t_img) # 提案手法によって生成されたセグメン画像
    accuracy = np.array(a_img) # 正解のセグメン画像

    # 適合率(precision)算出用変数
    back_all = 0            #全予測背景数
    nuclear_all = 0         #全予測核数
    cytoplasm_all = 0       #全予測細胞質数
    back_accuracy = 0       #予測背景数の中の正解数
    nuclear_accuracy = 0    #予測核数の中のの正解数
    cytoplasm_accuracy = 0  #予測細胞質数の中の正解数

    # 適合率(precision)算出
    
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
    
    # 再現率(recall)算出用変数
    back_all2 = 0            #全予測背景数
    nuclear_all2 = 0         #全予測核数
    cytoplasm_all2 = 0       #全予測細胞質数
    back_accuracy2 = 0       #予測背景数の中の正解数
    nuclear_accuracy2 = 0    #予測核数の中のの正解数
    cytoplasm_accuracy2 = 0  #予測細胞質数の中の正解数

    # 再現率(recall)算出
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if accuracy[i][j] == 0:  #正解が背景なら
                back_all2 += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が背景なら
                    back_accuracy2 += 1
            elif accuracy[i][j] == 1:  #正解が核なら
                nuclear_all2 += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が核なら
                    nuclear_accuracy2 += 1
            elif accuracy[i][j] == 2:  #正解が細胞質なら
                cytoplasm_all2 += 1
                if accuracy[i][j] == test[i][j]:  #正解と予測が細胞質なら
                    cytoplasm_accuracy2 += 1

    # 適合率算出
    back = back_accuracy/back_all                   #背景の適合率
    nuclear = nuclear_accuracy/nuclear_all          #核の適合率
    cytoplasm = cytoplasm_accuracy/cytoplasm_all    #細胞質の適合率

    # 再現率算出
    back2 = back_accuracy2/back_all2                   #背景の適合率
    nuclear2 = nuclear_accuracy2/nuclear_all2         #核の適合率
    cytoplasm2 = cytoplasm_accuracy2/cytoplasm_all2    #細胞質の適合率

    # F値算出
    f_measure_back = 2*back2*back / (back2+back) # 2*recall*precsion/recall+precision
    f_measure_nuclear = 2*nuclear2*nuclear / (nuclear2+nuclear) # 2*recall*precsion/recall+precision
    f_measure_cytoplasm = 2*cytoplasm2*cytoplasm / (cytoplasm2+cytoplasm) # 2*recall*precsion/recall+precision

    final_f_measure = (f_measure_back+f_measure_nuclear+f_measure_cytoplasm)/3 # F値のマクロ平均
    #print(args[1])
    #print("back:", back)
    #print("nuclear:", nuclear)
    #print("cytoplasm:", cytoplasm)
    print("F値:"+str(final_f_measure))
    #print("")
#------------------main--------------------#
args = sys.argv

img = Image.open("Generated_seg_images/seg_" + args[1])
accuracy_img = Image.open("SegmentationClass/" + args[1])

evalution(img, accuracy_img)
