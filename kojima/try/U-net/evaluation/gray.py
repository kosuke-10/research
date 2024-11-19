import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

red=0.5
green=0.5
blue=0


def main():
    if len(sys.argv) < 2:
        print("Usage: python your_program_name.py input_image_name.png/jpg")
        sys.exit(1)

    input_path = "Original/" + sys.argv[1]
    output_path = "OutGray/"+sys.argv[1]  # ここで出力パスを固定
    hist_path="Histgram/"+sys.argv[1]
    
    # 画像の読み込み
    img = Image.open(input_path)
    
    data = np.array(img)
    
    gg=img.convert("L")
    gg.save("OutGray/P"+sys.argv[1])

    data = np.array(img)
    RGgray = 0.5* data[:,:,0] + 0.5 * data[:,:,1] + 0 * data[:,:,2]
    RBgray = 0.5* data[:,:,0] + 0 * data[:,:,1] + 0.5 * data[:,:,2]
    
    # グレースケール画像をPIL Imageオブジェクトに変換
    RGgray_img = Image.fromarray(RGgray.astype(np.uint8))
    RBgray_img = Image.fromarray(RBgray.astype(np.uint8))
        
    RGgray_img.save("OutGray/RG"+sys.argv[1])
    RBgray_img.save("OutGray/RB"+sys.argv[1])
    
    gg_array = np.array(gg)  # PIL ImageをNumPyのndarrayに変換
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img_array = clahe.apply(gg_array)
    clahe_img = Image.fromarray(clahe_img_array)  # ndarrayをPIL Imageに変換
    clahe_img.save("OutGray/H" + sys.argv[1])
    
    # # ヒストグラムの取得
    # hist = img.histogram()
    # # ヒストグラムの描画
    # plt.bar(range(256), hist)
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram')

    # # ヒストグラムを画像ファイルとして保存
    # plt.savefig(hist_path, bbox_inches='tight')  # bbox_inchesは余白を削除するためのオプション

    

if __name__ == "__main__":
    main()
