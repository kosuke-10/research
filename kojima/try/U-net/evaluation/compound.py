from PIL import Image
import numpy as np
import sys

color_Palete = np.array([[[128, 0, 0],
                          [0, 128, 0],
                          [0, 0, 0]]])
    

def Get_mask(seg_image_path):
    
    # セグメンテーション画像を読み込む
    seg_img = Image.open(seg_image_path).convert("RGB")
    
    # 画像をnumpy配列に変換
    mask_array = np.array(seg_img)

    # mask_arrayの各ピクセルが[0,0,0]に該当するかどうかを判定
    black_mask = np.all(mask_array == [0, 0, 0], axis=-1)

    # 該当するピクセルを黒に、それ以外のピクセルを白にするマスクを作成
    result_mask = np.where(black_mask, 0, 255).astype(np.uint8)
    # 新しい配列を作成し、マスクがTrueの場所だけ255 (白) にする
    
    extracted_img = Image.fromarray(result_mask)

    # 画像を指定されたパスに保存
    extracted_img.save("cut_domain.png")
    
    return result_mask

def Get_domain(image_path,mask, save_path):
    seg_img = Image.open(image_path).convert("RGB")
    
    # 画像をnumpy配列に変換
    mask_array = np.array(seg_img)
    
    # 新しい画像配列を作成
    final_image = np.zeros_like(mask_array)

    # result_maskが白の領域（領域内）で、元の画像が[128,0,0]の部分はそのままの色にする
    red_domain = np.all(mask_array == [128, 0, 0], axis=-1)
    final_image[np.logical_and(mask == 255, red_domain)] = [128, 0, 0]

    # result_maskが白の領域（領域内）で、元の画像が[128,0,0]以外の部分は[0,128,0]にする
    not_red_domain = np.logical_not(red_domain)
    final_image[np.logical_and(mask == 255, not_red_domain)] = [0, 128, 0]

    # result_maskが黒の部分（領域外）は[0,0,0]にする
    final_image[mask == 0] = [0, 0, 0]

    # 新しい画像を保存
    Image.fromarray(final_image).save(save_path)

    return final_image


good_path= "goodseg/seg_" + sys.argv[1]
mix_path="mixseg/seg_"+sys.argv[1]

out_path="comseg/seg_"+ sys.argv[1]

mask=Get_mask(good_path)

Get_domain(mix_path,mask,out_path)


