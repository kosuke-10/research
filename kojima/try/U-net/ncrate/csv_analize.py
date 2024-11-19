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

# CSVファイルを読み込む
df = pd.read_csv('data.csv')

# DataFrameの内容を表示
print(df)

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