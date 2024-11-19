# python find_nc_rate.py 1.png
import sys
import io
from PIL import Image
import numpy  as np 
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
import cleanup
import pandas as pd


def conventional_nc(img_np):

    nuclear = 0
    cytoplasm = 0
    background = 0
    color_Palete = np.array([[[128, 0, 0],
                            [0, 128, 0],
                            [0, 0, 0]]])

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            if (img_np[i][j] == color_Palete[0][0]).all(): # nuclear:細胞核 赤色
                nuclear += 1
            elif (img_np[i][j] == color_Palete[0][1]).all(): # cytoplasm:細胞質 緑色
                cytoplasm += 1
            elif (img_np[i][j] == color_Palete[0][2]).all(): # background:背景 黒色
                background += 1 

    # for i in range(img_np.shape[0]):
    #     for j in range(img_np.shape[1]):
    #         if img_np[i][j] == 2:
    #             nuclear += 1
    #         elif img_np[i][j] == 1:
    #             cytoplasm += 1
    #         elif img_np[i][j] == 0:
    #             background += 1

    # np.set_printoptions(threshold=np.inf) # printで省略せず表示する場合コメントアウトOFF
    # print("check", color_Palete[0][0])
    # print("check", img_np[25][25])
    # print("check", img_np.shape[2])

    # print("n:", nuclear)
    # print("c:", cytoplasm)
    # print("b:", background)
    nc_rate = nuclear/(nuclear + cytoplasm)
    print("n/c比 =", nc_rate)

def individual_nc(img_np):
   # 核と細胞質の対応関係およびN/C比のデータを格納するためのリスト
   
   
    if '.jpg' in sys.argv[1]:
        output_name = sys.argv[1].replace(".jpg", "")
    elif '.jpeg' in sys.argv[1]:
        output_name = sys.argv[1].replace(".jpeg", "")
    elif '.png' in sys.argv[1]:
        output_name = sys.argv[1].replace(".png", "")
   
   
    image_number = []
    nucleus_list = []
    cytoplasm_list = []
    nc_ratio_list = []
    total_area_list = []

    #核と細胞質を対応付ける
    labeled,nucleus_to_cytoplasm_map = cleanup.define_relation(img_np)
    
    # ラベル統一後の各ラベル領域のプロパティを計算
    props = regionprops(labeled)
    # 各ラベルの面積（area）情報を取得
    areas = {prop.label: prop.area for prop in props}

    print("\n核と細胞質の面積比（N/C比）:")
    for nucleus_label in nucleus_to_cytoplasm_map:
        nucleus_area = areas.get(nucleus_label, 0)
        cytoplasm_label = nucleus_to_cytoplasm_map[nucleus_label][0]  # 対応する細胞質領域のラベルを取得
        cytoplasm_area = areas.get(cytoplasm_label, 0)  # 対応する細胞質領域の面積を取得
        total_area = nucleus_area + cytoplasm_area  # 核と細胞質の合計面積

        # N/C比の計算
        nc_ratio = nucleus_area / total_area
        # リストにデータを追加
        image_number.append(sys.argv[1])
        nucleus_list.append(nucleus_label)
        cytoplasm_list.append(cytoplasm_label)
        nc_ratio_list.append(nc_ratio)
        total_area_list.append(total_area)
        
        # DataFrameの作成
    df = pd.DataFrame({
        '核': nucleus_list,
        '細胞質': cytoplasm_list,
        'N/C比': nc_ratio_list,
        '合計面積':total_area_list
    })
    # DataFrameの表示
    
    print(df)
    return df, labeled
    

if __name__ == "__main__":
    
    # img = Image.open("Seg-11/seg_" + sys.argv[1])
    # img_np = np.array(img)  
    # individual_nc(img_np)
    

    img = Image.open("Seg-11/seg_" + sys.argv[1]).convert("RGB")
    img_np = np.array(img)
    conventional_nc(img_np)