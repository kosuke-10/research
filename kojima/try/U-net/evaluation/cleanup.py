# python find_nc_rate.py 1.png
import sys
import io
from PIL import Image
import numpy  as np 
from skimage.measure import label, regionprops_table,regionprops
from skimage.segmentation import find_boundaries
import pandas as pd


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

color_Palete = np.array([[[128, 0, 0],
                          [0, 128, 0],
                          [0, 0, 0]]])

#核領域のみに囲まれた背景領域を核領域に変更
def change_back(img_np):
    labeled = label(img_np == 0)
    properties = ['label', 'area']
    
    # 画像全体の面積（ピクセル数）を計算
    total_area = img_np.size
    
    # 各ラベルのプロパティを計算
    df = pd.DataFrame(regionprops_table(labeled, properties=properties))

    for label_id, area in zip(df['label'], df['area']):
        area_percentage = (area / total_area) * 100

        if area_percentage <= 1:
            # 背景ラベルの境界を取得
            background_boundary = find_boundaries(labeled == label_id, mode='outer')

            # 境界がすべて核領域に囲まれているかチェック
            if np.all(img_np[background_boundary] == 1):
                # 背景ラベル領域を核領域に変更
                img_np[labeled== label_id] = 1
    return img_np


#余分な細胞核領域の排除
def remove_wastenuclear(img_np):
    labeled = label(img_np == 1)
    properties = ['label', 'area']
    
    # 画像全体の面積（ピクセル数）を計算
    total_area = img_np.size
    # print(f"トータルピクセル数：{total_area}")
    
    df = pd.DataFrame(regionprops_table(labeled, properties=properties))
    
    # 各ラベル領域の面積（ピクセル数）をチェックし、小さい核領域を細胞質領域に変更
    for label_id, area in zip(df['label'], df['area']):
        area_percentage = (area / total_area) * 100
        # print(f"Label {label_id} has {area} pixels, which is {area_percentage:.2f}% of the image")
        if area_percentage <= 0.025:
            img_np[labeled == label_id] = 2  # 小さい核領域を細胞質領域に変更
    
    return img_np

#余分な細胞質領域の削除
def remove_wastecytoplasm(img_np):

    #核領域境界の外側と内側を取得    
    outer = find_boundaries(img_np == 1, mode='outer')
    inner = find_boundaries(img_np == 1, mode='inner')
    #重ね合わせて一つの境界化
    boundaries = outer | inner
    
    # 画像内の異なる領域にラベルを割り当て
    labeled_image = label(img_np)

    # 境界と重ならないラベル領域を削除(背景化)
    for region_label in np.unique(labeled_image):
        region_mask = labeled_image == region_label
        
        if not np.any(boundaries[region_mask]):
            img_np[region_mask] = 0  # 消去
    
    #核輪郭の表示用
    #img_np[boundaries]=5

    return img_np


def cleanup(img_np):
    seg_image = Image.open("Palette.png")  # カラーパレット取得用
    palette = seg_image.getpalette()
    
    cleaned_np = change_back(remove_wastecytoplasm(remove_wastenuclear(img_np)))
    
    cleaned_pil = Image.fromarray(np.uint8(cleaned_np))
    cleaned_pil.putpalette(palette)
    cleaned_pil.save("cleanup/seg_"+sys.argv[1])
    
    return cleaned_np
    

def define_relation(img_np):
    
    seg_image = Image.open("Palette.png")  # カラーパレット取得用
    palette = seg_image.getpalette()

    # 核と細胞質領域の識別
    nuclei = img_np == 1
    cytoplasm = img_np == 2
    back = img_np == 0

    labeled=label(img_np)

    # 核と細胞質の関連を格納する辞書
    nucleus_to_cytoplasm_map = {}
    cytoplasm_to_nucleus_map = {}
    
    # ラベリングされた各領域に対してループ
    for label_number in np.unique(labeled):
             
        # 核領域に対応するラベルのみ処理
        if np.any(nuclei[labeled == label_number]):
            
            #核領域の境界外側を出力
            nucleus_boundary = find_boundaries(labeled == label_number, mode='outer')

            # 境界に重なる細胞質領域を特定
            overlapping_labels = np.unique(labeled[nucleus_boundary])
            
            # 核に重なるすべての細胞質領域を記録（リストとして）
            nucleus_overlaps = [label for label in overlapping_labels if label != label_number and label != 0]
            nucleus_to_cytoplasm_map[label_number] = nucleus_overlaps

            # 細胞質領域ごとに重なる核のラベルをリストとして記録
            for cyto_label in nucleus_overlaps:
                if cyto_label not in cytoplasm_to_nucleus_map:
                    cytoplasm_to_nucleus_map[cyto_label] = []
                cytoplasm_to_nucleus_map[cyto_label].append(label_number)


    for _ in range(2):  # 2回繰り返す
        # 核にまたがる複数の細胞質領域のラベルを統合
        for nucleus_label, cytoplasm_labels in list(nucleus_to_cytoplasm_map.items()):
            if len(cytoplasm_labels) > 1:
                unified_cytoplasm_label = min(cytoplasm_labels)
                for cyto_label in cytoplasm_labels:
                    if cyto_label != unified_cytoplasm_label:
                        labeled[labeled == cyto_label] = unified_cytoplasm_label
                        # cytoplasm_to_nucleus_mapの更新
                        if cyto_label in cytoplasm_to_nucleus_map:
                            # unified_cytoplasm_label が辞書に存在しない場合は新しいリストを作成
                            if unified_cytoplasm_label not in cytoplasm_to_nucleus_map:
                                cytoplasm_to_nucleus_map[unified_cytoplasm_label] = []
                            for nuc_label in cytoplasm_to_nucleus_map[cyto_label]:
                                cytoplasm_to_nucleus_map[unified_cytoplasm_label].append(nuc_label)
                            del cytoplasm_to_nucleus_map[cyto_label]
                nucleus_to_cytoplasm_map[nucleus_label] = [unified_cytoplasm_label]

        # 同じ細胞質領域に属する核のラベルを統一
        for cytoplasm_label, nucleus_labels in list(cytoplasm_to_nucleus_map.items()):
            unified_nucleus_label = min(nucleus_labels)
            for nucleus_label in nucleus_labels:
                if nucleus_label != unified_nucleus_label:
                    labeled[labeled == nucleus_label] = unified_nucleus_label
                    # nucleus_to_cytoplasm_mapの更新
                    if nucleus_label in nucleus_to_cytoplasm_map:
                        # unified_nucleus_label が辞書に存在しない場合は新しいリストを作成
                        if unified_nucleus_label not in nucleus_to_cytoplasm_map:
                            nucleus_to_cytoplasm_map[unified_nucleus_label] = []
                        nucleus_to_cytoplasm_map[unified_nucleus_label].extend(nucleus_to_cytoplasm_map[nucleus_label])
                        del nucleus_to_cytoplasm_map[nucleus_label]
            cytoplasm_to_nucleus_map[cytoplasm_label] = [unified_nucleus_label]


    print("\n核と細胞質の対応関係:")
    for nucleus_label, cytoplasm_labels in nucleus_to_cytoplasm_map.items():
        print(f"核 {nucleus_label} は次の細胞質領域と関連しています: {cytoplasm_labels}")
    
    for  cytoplasm_label,nucleus_labels in cytoplasm_to_nucleus_map.items():
          print(f"細胞質 {cytoplasm_label} は次の核領域と関連しています: {nucleus_labels}")

    # # ラベル変更後の画像をPIL画像に変換
    labeled_image_pil = Image.fromarray(np.uint8(labeled))
    if labeled_image_pil.mode != 'P':
        labeled_image_pil = labeled_image_pil.convert('P')
    labeled_image_pil.putpalette(palette)
    labeled_image_pil.putpalette(palette)
    labeled_image_pil.save("Labeled/seg_" + sys.argv[1])
            
    # 画像をファイルとして保存
    return labeled,nucleus_to_cytoplasm_map
    
def main():
   
    args = sys.argv

    img = Image.open("Seg-val/seg_" + args[1])
    img_np = np.array(img)
    
    cleaned = cleanup(img_np)

        
    division,map = define_relation(cleaned)
  

if __name__ == "__main__":
    main()
    