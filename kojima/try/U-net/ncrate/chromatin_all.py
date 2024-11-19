from PIL import Image
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from natsort import natsorted
from skimage.measure import label,regionprops
import find_nc_rate_all as nc_rate
import pandas as pd
import cleanup_all as cleanup
import cal_method as cal
import matplotlib.pyplot as plt


# 表示する最大行数を設定（例えば100行）
pd.set_option('display.max_rows', 300)
    

#nuclear領域のヒストグラムを作成
def nuclear_histgram(df,wsi_name):
    
    image_name = df['img_name']
    nucleus_label = df['nuclear']
    
    #名前などの取得
    color_name = image_name.replace('seg_','')
    color_dir = "color-" + wsi_name
    labeled = "Labeled"

    # 対応する画像を読み込む
    img = Image.open(f"{color_dir}/{color_name}")
    img_np = np.array(img)

    # 対応するラベルセグメント画像を読み込む 
    labeled_img = Image.open(f"{labeled}/{image_name}")
    labeled_img_np = np.array(labeled_img)   
    
    # nuclearラベルに対応する領域を取得
    nucleus_region = labeled_img_np == nucleus_label

    # nuclear領域を抽出してグレースケール化
    nucleus_region_image = img_np * np.expand_dims(nucleus_region, axis=-1)
    # マスクの外部（nuclear領域以外）を白色（255）に設定
    nucleus_region_image[nucleus_region == 0] = 255
    
    nucleus_pil = Image.fromarray(nucleus_region_image).convert('L')

    # ファイル名と拡張子を分離
    name, ext = os.path.splitext(image_name)
    save_name = f"{df['type']}_{name}_{nucleus_label}{ext}"
    
    # グレースケール化されたnuclear領域の画像を保存
    nucleus_pil.save("domain/" +save_name)
    
    nucleus_np = np.array(nucleus_pil)
    
    histgram = calc_hist(nucleus_np,save_name)
    
    return histgram

#基本的なヒストグラムの作成と正規化
def calc_hist(img_np,save_name):
    #背景を白⇒255は除外
    
    histogram, _ = np.histogram(img_np, bins=255, range=(0, 254))
    # histogram, _ = np.histogram(data, bins=256, range=(0, 255))
    histogram=histogram / histogram.sum()  # 正規化
    save_hist(histogram,save_name)
    return histogram

#ヒストグラムの保存
def save_hist(hist,save_name):
    plt.plot(hist, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
        
    plt.savefig("Histgram/"+save_name)
    plt.close()


#陰性確実と疑いのnuclearヒストグラム作成⇒陰性の平均と疑いのリスト
def gather_histdata(df,wsi_name):
    
    cleanup.manage_directory("domain")
    cleanup.manage_directory("Histgram")

    negatives = []  #単純なリスト
    suspects = []   #辞書型のリスト
    
    # negative+ タイプの行を抽出
    negative_plus_df = df[df['type'] == 'negative+']
    suspect_df = df[df['type'] == 'suspect']
    
    # 各行に対して処理
    for index, row in negative_plus_df.iterrows():
        histgram = nuclear_histgram(row,wsi_name)
        negatives.append(histgram)
        
    ave_negative = np.mean(negatives, axis=0)
    save_hist(ave_negative,"average")
        
    for index, row in suspect_df.iterrows():
        histgram = nuclear_histgram(row,wsi_name)
        # img_nameとnuclearの情報をヒストグラムと一緒に辞書として保存
        hist_info = {
            'img_name': row['img_name'],
            'nuclear': row['nuclear'],
            'histgram': histgram,
            # 'truth':row['truth']
        }
        suspects.append(hist_info)
        
    
    return ave_negative,suspects

#クロマチン増加
def hyperchromasia(df,wsi_name):
    
    ave_negative,suspects = gather_histdata(df,wsi_name)
    
    for suspect in suspects:
        suspect_hist = suspect['histgram']
        kl_div = cal.kl_divergence(ave_negative,suspect_hist)
        js_div = cal.js_divergence(ave_negative,suspect_hist)
        skew = cal.calculate_skewness(suspect_hist)
        c_skew = cal.calculate_custom_skewness(suspect_hist)
        print(f"画像名:{suspect['img_name']}  核ラベル:{suspect['nuclear']}  kl_div:{kl_div}  js_div:{js_div} skew:{skew}  c_skew:{c_skew}" )    
    
    # skew_ave = cal.calculate_skewness(ave_negative)
    # cus_ave = cal.calculate_custom_skewness(ave_negative)
    # print(skew_ave,cus_ave)
    # calculate_js_divergence(suspects,ave_negative)
    

def nuclear_irregularity(df):
    new_rows = []

    cleanup.manage_directory("cir_nuclear")

    suspect_df = df[df['type'] == 'suspect']
    for index, row in suspect_df.iterrows():
        img_path = "Labeled/" + row['img_name']
        labeled = Image.open(img_path)
        labeled_np = np.array(labeled)

        nuclear_region = labeled_np == row['nuclear']
        labeled_region = label(nuclear_region)

        # 画像の表示用に設定
        fig, ax = plt.subplots()
        ax.imshow(labeled_region, cmap="turbo", alpha=0.5)
        ax.axis('off')  # 軸の表示をオフ

        for region in regionprops(labeled_region):
            area = region.area
            perimeter = region.perimeter_crofton
            circularity = 4 * np.pi * area / (perimeter**2)

            new_row = row.copy()
            new_row['nu_indiv'] = region.label
            new_row['circ'] = circularity
            new_rows.append(new_row)  

            # 円形度を画像上に表示
            y, x = region.centroid
            ax.text(x, y, np.round(circularity, 2), color='black', fontsize=8, ha='center', va='center')

        #余白を排除して画像保存
        plt.tight_layout()
        plt.savefig("cir_nuclear/" + row['img_name'] + '_' + str(row['nuclear']) + '.png')
        plt.close()

    new_df = pd.DataFrame(new_rows)
    new_df.reset_index(drop=True, inplace=True)

    print(new_df)

    # properties = ['label','area','centroid',"major_axis_length","minor_axis_length",'perimeter_crofton']
    # new = pd.DataFrame(regionprops_table(label_image,properties=properties))
        
def calculate_js_divergence(test_data, ave_negative):
    # JS分岐の計算と真のラベルの記録
    js_divergences = []
    for sample in test_data:
        histgram = sample['histgram']
        js_div = cal.js_divergence(ave_negative, histgram)
        js_divergences.append((js_div, sample['truth']))  # ('js_div', 'negative' or 'positive')
    
    # 閾値のリストを定義（例：0から最大JS分岐までの範囲で0.01刻み）
    thresholds = np.arange(0, max(js_div[0] for js_div in js_divergences), 0.01)
    
    # 各閾値に対して分類精度を計算
    accuracies = []
    for threshold in thresholds:
        # 'unknown'でないサンプルのみをフィルタリング
        filtered_js_divergences = [(js_div, truth) for js_div, truth in js_divergences if truth != 'unknown']

        # 正しい分類の数をカウント
        correct_classifications = sum(
            1 for js_div, truth in filtered_js_divergences if (js_div < threshold and truth == 'negative') or (js_div >= threshold and truth == 'positive')
        )
        
        # 'unknown'でないサンプルの総数
        total_samples = len(filtered_js_divergences)
        
        # 精度を計算
        accuracy = correct_classifications / total_samples if total_samples > 0 else 0
        accuracies.append(accuracy)

    # 最も高い分類精度を与える閾値を見つける
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = max(accuracies)

    # 閾値と精度のグラフをプロット
    plt.figure()
    plt.plot(thresholds, accuracies)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Threshold vs. Accuracy')
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best threshold = {best_threshold}, Accuracy = {best_accuracy}')
    plt.legend()
    plt.savefig("graph.png")

    print(best_threshold,best_accuracy)

# def classify_chromatin(df):

def nuclear_circularity(df):
    new_rows = []

    cleanup.manage_directory("cir_nuclear")

    for index, row in df.iterrows():
        img_path = "Labeled/" + row['img_name']
        labeled = Image.open(img_path)
        labeled_np = np.array(labeled)

        nuclear_region = labeled_np == row['nuclear']
        labeled_region = label(nuclear_region)

        # 画像の表示用に設定
        fig, ax = plt.subplots()
        ax.imshow(labeled_region, cmap="turbo", alpha=0.5)
        ax.axis('off')  # 軸の表示をオフ

        for region in regionprops(labeled_region):
            area = region.area
            perimeter = region.perimeter_crofton
            circularity = 4 * np.pi * area / (perimeter**2)

            new_row = row.copy()
            new_row['nu_indiv'] = region.label
            new_row['circ'] = circularity
            new_rows.append(new_row)  

            # 円形度を画像上に表示
            y, x = region.centroid
            ax.text(x, y, np.round(circularity, 2), color='black', fontsize=8, ha='center', va='center')

        #余白を排除して画像保存
        plt.tight_layout()
        plt.savefig("cir_nuclear/" + os.path.splitext(row['img_name'])[0]+ '_' + str(row['nuclear']) + '.png')
        plt.close()

    new_df = pd.DataFrame(new_rows)
    new_df.reset_index(drop=True, inplace=True)

    print(new_df)


def chromatin():

    wsi_name = sys.argv[1]
    
    df = nc_rate.individual_nc(wsi_name)

#陰性確実と疑いのnuclearヒストグラム作成⇒陰性の平均と疑いをリスト形式で
    hyperchromasia(df,wsi_name)
    
    # nuclear_irregularity(df)

    nuclear_circularity(df)

if __name__ == "__main__":
    chromatin()

