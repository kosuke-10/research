

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
from scipy import stats

#ヒストグラムの作成と正規化
def calc_hist(img_path):
    img = Image.open(img_path)
    data = np.array(img.convert("L"))
    histogram, _ = np.histogram(data, bins=255, range=(0, 254))
    # histogram, _ = np.histogram(data, bins=256, range=(0, 255))
    histogram=histogram / histogram.sum()  # 正規化
    save_hist(histogram,img_path)
    return histogram

def save_hist(hist,img_path):
    plt.plot(hist, color='blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.savefig("Histgram01/nor_"+os.path.basename(img_path))
    plt.close()

#KL-divergenceの計算
def kl_divergence(p, q):
    # 0で割ることを避けるために非常に小さな値を加える
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    # KL-divergenceを計算
    return np.sum(p * np.log(p / q))

#ジェンセン・シャノン発散の計算
def js_divergence(p, q):
    m = (p + q) / 2
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    return (kl_pm + kl_qm) / 2

#単独のエントロピーの計算
def entropy(histgram):

    # ゼロでない確率値のみを対象とする
    histgram = histgram[histgram > 0]

    # エントロピーの計算
    entropy = -np.sum(histgram * np.log2(histgram))
    return entropy

def calculate_spread(histogram):
    # 分散と標準偏差の計算
    variance = np.var(histogram)
    std_deviation = np.sqrt(variance)
    return variance, std_deviation

def calculate_kurtosis(histogram):
    # 尖度の計算
    kurtosis = stats.kurtosis(histogram, fisher=False)
    return kurtosis

def calculate_iqr(histogram):
    # 四分位範囲（IQR）の計算
    q75, q25 = np.percentile(histogram, [75 ,25])
    iqr = q75 - q25
    return iqr

def calculate_skewness(histogram):
    # ヒストグラムのビンの中心値を計算
    bin_centers = np.arange(len(histogram))
    # 平均と標準偏差
    mean = np.average(bin_centers, weights=histogram)
    std_dev = np.sqrt(np.average((bin_centers - mean)**2, weights=histogram))
    # 歪度の計算
    skewness = np.average((bin_centers - mean)**3, weights=histogram) / std_dev**3
    return skewness

def calculate_custom_skewness(histogram):

    """
    Calculate custom skewness of a histogram based on mode with fixed bin centers.

    :param histogram: np.array, the values of the histogram (frequencies)
    :return: float, the custom skewness value based on the mode
    """
    # Fixed bin centers for the specified histogram settings


    bin_centers = np.linspace(0, 254, 255)
    # Find the mode value index
    mode_index = np.argmax(histogram)
    mode_value = bin_centers[mode_index]

    # Calculate differences from the mode
    differences = bin_centers - mode_value

    # Calculate weighted sum of cubed differences
    weighted_diff_cubed_sum = np.sum(histogram * differences**3)

    # Calculate pseudo standard deviation
    weighted_diff_squared_sum = np.sum(histogram * differences**2)
    pseudo_std_dev = np.sqrt(weighted_diff_squared_sum / np.sum(histogram))

    # Avoid division by zero
    if pseudo_std_dev == 0:
        return 0

    # Calculate custom skewness
    custom_skewness = weighted_diff_cubed_sum / (pseudo_std_dev**3 * np.sum(histogram))

    return custom_skewness

# 使用例:
# histogram はヒストグラムの頻度データとして定義される
# custom_skew = calculate_fixed_custom_skewness(histogram)
# print("Custom skewness:", custom_skew)



def main():
# ネガティブ（正常な）細胞の画像群から平均ヒストグラムを計算
    # negative_histograms = [calc_hist(Image.open(image_path)) for image_path in negative_image_paths]
    # average_negative_histogram = np.mean(negative_histograms, axis=0)

    # # 検査対象画像のヒストグラムを計算
    # test_histogram = calc_hist(Image.open(test_image_path))

    # # KL-divergenceを計算
    # kl_div = calculate_kl_divergence(test_histogram, average_negative_histogram)
    
    img_path1="domain/"+sys.argv[1]
    img_path2="domain/"+sys.argv[2]
    
    hist_p=calc_hist(img_path1)
    hist_q=calc_hist(img_path2)
    
    # kl_div=kl_divergence(hist_p,hist_q)
    jensen=js_divergence(hist_p,hist_q)    
    print(f"JS-divergergence:{jensen}")
    
    entropy_p=entropy(hist_p)
    entropy_q=entropy(hist_q)
    
    print(f"entropy1:{entropy_p}")
    print(f"entropy2:{entropy_q}")
    
    var1,std1=calculate_spread(hist_p)
    var2,std2=calculate_spread(hist_q)
    print(f"分散1:{var1},標準偏差1:{std1}")
    print(f"分散2:{var2},標準偏差2:{std2}")
    
    kurt1=calculate_kurtosis(hist_p)
    kurt2=calculate_kurtosis(hist_q)
    
    iqr1=calculate_iqr(hist_p)
    iqr2=calculate_iqr(hist_q)
    
    print(f"尖度1:{kurt1},四分位範囲1:{iqr1}")
    print(f"尖度2:{kurt2},四分位範囲2:{iqr2}")
 


if __name__ == "__main__":
    main()
