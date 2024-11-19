from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語をサポートするフォントに変更する
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


color_Palete = np.array([[[128, 0, 0],
                          [0, 128, 0],
                          [0, 0, 0]]])
    

def Get_mask(seg_image_path):
    
    # セグメンテーション画像を読み込む
    seg_img = Image.open(seg_image_path).convert("RGB")
    
    # 画像をnumpy配列に変換
    mask_array = np.array(seg_img)

    # mask_arrayの各ピクセルが[0,0,0]に該当するかどうかを判定
    map = np.all(mask_array == color_Palete[0][0], axis=-1)


    mask_l = np.where(map, 255, 0).astype(np.uint8)

    # Convert to a 3-channel RGB image
    mask = np.stack([mask_l]*3, axis=-1)

    # 新しい配列を作成し、マスクがTrueの場所だけ255 (白) にする
    extracted_img = Image.fromarray(mask)

    # 画像を指定されたパスに保存
    # extracted_img.save("cut_domain.png")
    
    return mask_l

def Get_domain(image_path,mask,out_path):

    img = Image.open(image_path)
    
    # 画像をnumpy配列に変換
    img_array = np.array(img)
    
    extracted_array = np.where(mask[:, :, None] == 255, img_array, 255)

    extracted=Image.fromarray(extracted_array)
    
    extracted.save(out_path)
    return extracted

def save_rgb_histogram(image, save_path):
    # 画像を読み込み、それをNumPy配列に変換

    data = np.array(image)

    # R, G, Bの各チャンネルに対してヒストグラムを計算
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        histogram, bin_edges = np.histogram(data[:, :, i], bins=255, range=(0, 254))
        plt.plot(bin_edges[0:-1], histogram, color=color)

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(['Red', 'Green', 'Blue'])
    plt.title('RGB Histogram')

    # Save the plot as an image
    plt.savefig(save_path)
    plt.close()
    
def save_grayscale_histogram(img, save_path):
    
    # カラー画像をグレースケールに変換
    grayscale_img = img.convert("L")
    grayscale_img.save("graynuclear/"+sys.argv[1])
    data = np.array(grayscale_img)

    # グレースケール画像のヒストグラムを計算
    histogram, bin_edges = np.histogram(data, bins=255, range=(0, 254))
    plt.plot(bin_edges[0:-1], histogram, color='black')

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Grayscale Histogram')

    # Save the plot as an image
    plt.savefig(save_path)
    plt.close()

# For testing purposes, you can use this function with valid 'image_path' and 'save_path' in your local environment.
# save_grayscale_histogram(image_path, save_path)

def save_combined_grayscale_histogram(img1, img2, save_path):
    # 画像1をグレースケールに変換し、numpy配列に変換
    grayscale_img1 = img1.convert("L")
    data1 = np.array(grayscale_img1)

    # 画像2をグレースケールに変換し、numpy配列に変換
    grayscale_img2 = img2.convert("L")
    data2 = np.array(grayscale_img2)

    # 画像1のグレースケール画像のヒストグラムを計算
    histogram1, bin_edges1 = np.histogram(data1, bins=255, range=(0, 254))

    # 画像2のグレースケール画像のヒストグラムを計算
    histogram2, bin_edges2 = np.histogram(data2, bins=255, range=(0, 254))

    # ヒストグラムをプロット
    plt.figure()
    plt.plot(bin_edges1[0:-1], histogram1, color='blue', label="良性")
    plt.plot(bin_edges2[0:-1], histogram2, color='red', label="悪性")

    # プロットのラベルとタイトルを設定
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(['良性', '悪性'], fontsize=50)

    # プロットを画像として保存
    plt.savefig(save_path)
    plt.close()

def calculate_average(img):
    # 画像を読み込み、グレースケールに変換し、NumPy配列に変換
    grayscale_img = img.convert("L")

    data = np.array(grayscale_img)
    
    # ピクセル値が255以外の箇所を取得
    non_255_values = data[data != 255]
    
    # 平均を計算
    average = np.mean(non_255_values)
    
    print(sys.argv[1]+" 平均値:"+str(average))
    
    return average


def main():

    seg_path1="mixsegc/seg_"+sys.argv[1]
    ori_path1="Original/"+sys.argv[1]
    
    seg_path2="mixsegc/seg_"+sys.argv[2]
    ori_path2="Original/"+sys.argv[2]
    out_path="domain/"+ sys.argv[1]
    
    hist_path="Histgram/"+ sys.argv[1]
    ghist_path="Histgram01/g_"+ sys.argv[1]
    double_path="Histdouble/"+ sys.argv[1]

    mask1=Get_mask(seg_path1)
    mask2=Get_mask(seg_path2)
    extracted1=Get_domain(ori_path1,mask1,out_path)
    extracted2=Get_domain(ori_path2,mask2,out_path)
    
    save_combined_grayscale_histogram(extracted1,extracted2,double_path)
    
    # save_rgb_histogram(extracted,hist_path)
    
    # save_grayscale_histogram(extracted,ghist_path)
    
    # calculate_average(extracted)


if __name__ == "__main__":
    main()

