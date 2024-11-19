from PIL import Image
import numpy as np
import sys

def get_nucleus_color_from_segmentation(segmented_image, palette, nucleus_index=1):
    """セグメンテーション画像から細胞核の色を取得する関数"""
    color_index = np.unique(segmented_image)
    if nucleus_index in color_index:
        nucleus_color = palette[nucleus_index*3:nucleus_index*3+3]
        return nucleus_color
    else:
        return None

def extract_and_average(segmentation_path, grayscale_path, nucleus_index=1):
    # 画像を読み込む
    segmentation_img = Image.open(segmentation_path)
    grayscale_img = Image.open(grayscale_path)

    # 画像をnumpy配列に変換
    segmentation_arr = np.array(segmentation_img)
    grayscale_arr = np.array(grayscale_img)

    # セグメンテーション画像がマルチチャンネルの場合、グレースケールに変換
    if len(segmentation_arr.shape) == 3:
        segmentation_arr = segmentation_arr[:,:,0]

    # 細胞核のインデックスを持つピクセルの位置を見つける
    mask = (segmentation_arr == nucleus_index)

    # このマスクを使用して、グレースケール画像の該当するピクセルの値を取得
    extracted_values = grayscale_arr[mask]

    # 平均値を計算
    average_value = np.mean(extracted_values)

    return average_value


segmentation_path = "Generated_seg_images/seg_" + sys.argv[1]
grayscale_path= "JPEGImages/"+sys.argv[1]  # ここで出力パスを固定
seg_image = Image.open("Palette.png")  # カラーパレット取得用
palette = seg_image.getpalette()

segmentation_img = Image.open(segmentation_path)

# 細胞核の色をセグメンテーション画像から取得
nucleus_color = get_nucleus_color_from_segmentation(np.array(segmentation_img), palette)
if nucleus_color:
    print(f"Color of the nucleus region: {nucleus_color}")
    average_value = extract_and_average(segmentation_path, grayscale_path, nucleus_color)
    print(f"Average value of the nucleus region in the grayscale image: {average_value}")
else:
    print("Nucleus region not found in the segmentation image.")
