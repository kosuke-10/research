# python find_nc_rate.py 1.png
import sys
from natsort import natsorted
import os
from PIL import Image
import numpy  as np 
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
import cleanup_all as cleanup
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
args = sys.argv

# 最大表示行数の設定（Noneにすると全行表示）
pd.set_option('display.max_rows', None)

# 最大表示列数の設定（Noneにすると全列表示）
pd.set_option('display.max_columns', None)

def conventional_nc(img_np):

    nuclear = 0
    cytoplasm = 0
    background = 0
    color_Palete = np.array([[[128, 0, 0],
                            [0, 128, 0],
                            [0, 0, 0]]])

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            if (img_np[i][j] == color_Palete[0][0]).all(): # nuclear:細胞nuclear 赤色
                nuclear += 1
            elif (img_np[i][j] == color_Palete[0][1]).all(): # cytoplasm:cytoplasm 緑色
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
    # print("nc_ratio =", nc_rate)
    return nc_rate


def individual_nc(wsi_name):

    directory = "Seg-"+wsi_name
    
    #ディレクトリの初期化
    cleanup.manage_directory("Labeled")
    cleanup.manage_directory("cleanup")
    
    image_number = []
    nucleus_list = []
    cytoplasm_list = []
    nc_ratio_list = []
    total_area_list = []    

    for filename in natsorted(os.listdir(directory)):   #自然順序ソートして実行
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 画像ファイルの拡張子を確認
            image_path = os.path.join(directory, filename)
               
            #nuclearとcytoplasmを対応付ける
            labeled,nucleus_to_cytoplasm_map = cleanup.clean_and_relation(image_path)
    
            # ラベル統一後の各ラベル領域のプロパティを計算
            props = regionprops(labeled)
            # 各ラベルの面積（area）情報を取得
            areas = {prop.label: prop.area for prop in props}

            for nucleus_label in nucleus_to_cytoplasm_map:
                nucleus_area = areas.get(nucleus_label, 0)
                cytoplasm_label = nucleus_to_cytoplasm_map[nucleus_label][0]  # 対応するcytoplasm領域のラベルを取得
                cytoplasm_area = areas.get(cytoplasm_label, 0)  # 対応するcytoplasm領域の面積を取得
                total_area = nucleus_area + cytoplasm_area  # nuclearとcytoplasmのtotal_area

                # nc_ratioの計算
                nc_ratio = nucleus_area / total_area
                # リストにデータを追加
                image_number.append(filename)
                nucleus_list.append(nucleus_label)
                cytoplasm_list.append(cytoplasm_label)
                nc_ratio_list.append(nc_ratio)
                total_area_list.append(total_area)
        
    # DataFrameの作成
    df = pd.DataFrame({
        'img_name':image_number,
        'nuclear': nucleus_list,
        'cytoplasm': cytoplasm_list,
        'nc_ratio': nc_ratio_list,
        'total_area':total_area_list,
        'type':"default",
    })

    df = classify_nc(df)

    # Excelファイルを読み込む
    # truth_df = pd.read_excel(wsi_name+'.xlsx', header=None, names=['img_name', 'truth'])

    # # Excelファイルから読み込んだデータフレームと既存のデータフレームをマージする
    # # これは 'img_name' 列を基にして行われます
    # df = df.merge(truth_df, on='img_name', how='left')

    print(df)
    
    return df

#陰性などの分類（第一段階）
def classify_nc(df):
    # 画像ごとにnc_ratioが0.3を超える行が存在するかチェック
    over_threshold = df.groupby('img_name')['nc_ratio'].apply(lambda x: any(x > 0.3))

    # nc_ratioが0.3を超える行が存在する画像のリストを取得
    images_to_exclude = over_threshold[over_threshold].index.tolist()

    # nc_ratioが0.3を超える行が存在する画像を 'negative' に設定
    df.loc[(df['img_name'].isin(images_to_exclude)) & (df['nc_ratio'] <0.3), 'type'] = 'negative'

    # nc_ratioが0.3未満の行を 'negative+' に設定
    df.loc[(~df['img_name'].isin(images_to_exclude)) & (df['nc_ratio'] < 0.3), 'type'] = 'negative+'

    # nc_ratioが0.35以上の行を 'suspect' に再設定（既存の値を上書き）
    df.loc[df['nc_ratio'] >= 0.3, 'type'] = 'suspect'

    return df

def conventional_nc_all(args):
    
    # コマンドラインからディレクトリのリストを取得（スクリプト名は除く）
    directories = sys.argv[1:]   
    
    data = {
    'dir_name': [],       # 画像番号や名前を格納する空リスト
    'img_name': [],        # 核のデータを格納する空リスト
    'nc_ratio': [],       # 核と細胞質の比率を格納する空リスト
    }
    
    for directory in directories:
        for filename in natsorted(os.listdir(directory)):   #自然順序ソートを行う
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # 画像ファイルの拡張子を確認
                image_path = os.path.join(directory, filename)
                img = Image.open(image_path).convert("RGB")
                img_np = np.array(img)
                
                data['dir_name'].append(directory)
                data['img_name'].append(os.path.splitext(filename)[0]) 
                data['nc_ratio'].append(conventional_nc(img_np))
    
    df = pd.DataFrame(data)  # 変更後のデータでDataFrameを更新
    print(df)
    
    # CSVファイルとして保存
    df.to_csv('data.csv', index=False)
    
    # ヒストグラムの作成と保存
    plt.figure(figsize=(12, 6))
    for label, group_df in df.groupby('dir_name'):
        sns.histplot(group_df['nc_ratio'], bins=10, label=label, kde=False, alpha=0.6)
    plt.legend()
    plt.title('Histogram of NC Ratios by Directory Name')
    plt.xlabel('NC Ratio')
    plt.ylabel('Frequency')
    plt.savefig('histogram_nc_ratios.png')  # PNG形式で保存
    plt.close()  # プロットをクローズ


    # 密度プロットの作成と保存
    plt.figure(figsize=(12, 6))
    for label, group_df in df.groupby('dir_name'):
        sns.kdeplot(group_df['nc_ratio'], label=label)
    plt.legend()
    plt.title('Density Plot of NC Ratios by Directory Name')
    plt.xlabel('NC Ratio')
    plt.ylabel('Density')
    plt.savefig('density_plot_nc_ratios.png')  # PNG形式で保存
    plt.close()  # プロットをクローズ
    
    

if __name__ == "__main__":
        
    # wsi = sys.argv[1]
    # individual_nc(wsi)
            

    # img = Image.open("cleanup-T001/seg_" + args[1]).convert("RGB")
    # img_np = np.array(img)
    # conventional_nc()
    start_time = time.time()  # 実行開始時刻
    conventional_nc_all(args)
    
    end_time = time.time()  # 実行終了時刻
    elapsed_time = end_time - start_time  # 経過時間を計算

    print(f"Program took {elapsed_time} seconds to run.")