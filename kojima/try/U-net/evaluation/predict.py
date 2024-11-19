#####################################
# python3 predict.py sample.jpg
#####################################

import sys, time
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import skimage
from skimage.measure import label, regionprops_table
from skimage.draw import random_shapes
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt

def calculate_circularity(train_set, output_name, index_void=None): # train_set = (32, 256 ,256 ,4) ← model_unet.outputs
    train_image = train_set # Numpy → pillow # train_image.saveで画像を1枚ずつ保存可能
    # print("train_image",train_image) # train_image <PIL.Image.Image image mode=P size=256x256 at 0x7FE6F03666D8> # Image型
    # train_image.save("../mnt/kojima/train_image.png")

    train_image_array = np.asarray(train_image) # pillow → Numpy
    copy_image = np.copy(train_image_array) # WRITEABLE : False → True
    # print("train_image_array",train_image_array) # train_image_array [[0 2 0 ... 0 2 2] # numpy型
    Average_value(copy_image, output_name)

# ----------
# Average_value関数:1枚の画像の円形度の平均を求める関数
# 返り値:int型
# ----------
def Average_value(train_image_array, output_name): # 1枚の画像の円形度の平均を求める関数
    # print(train_image_array.shape)
    
    # 画像の2値化
    circularity_list = []
    for i in range(train_image_array.shape[0]):
        for u in range(train_image_array.shape[1]):
            if (train_image_array[i][u] == 1).any():
                train_image_array[i][u] = 1 # 細胞核:赤
            else:
                train_image_array[i][u] = 0 # 背景と細胞質:黒

    # #2値化象のラベリング
    label_image = label(train_image_array)
    label_image_f =label_image.astype(float)
    label_image_f[label_image_f==0]=np.nan
    
    #regionprops_tableによるラベリング領域の情報の取得
    properties = ['label','area','centroid',"major_axis_length","minor_axis_length",'perimeter_crofton']
    df = pd.DataFrame(regionprops_table(label_image,properties=properties))

    #円形度を求める
    df["circularity"]=4*np.pi*df["area"]/(df["perimeter_crofton"]**2)

    #ラベリング領域の情報をhtml形式で保存
    #df.to_html('rs_label_result.html')

    #円形度を画像上に表示
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(label_image_f,cmap="turbo",alpha=0.5)
    for n in df.index.values:
        ax.text(df["centroid-1"][n], df["centroid-0"][n],np.round(df["circularity"][n],2)) # ax.text(x,y,"文字列",size)
        circularity_list.append(np.round(df["circularity"][n],2))
    ax.set_title("Circularity")       
    plt.savefig("./Generated_cir_images/seg_circu_" + output_name + ".png") # matplotlibで画像を保存
    plt.close()
    
    """
    print("df.index.values",df.index.values) # df.index.values [0 1 2 3 4 5 6]
    print("circularity_list",circularity_list) # circularity_list [0.48, 0.08, 0.5, 1.75, 1.75, 1.75, 1.75]
    print("list_average",list_average) # list_average 1.1514285714285715
    print("round_abs_list_average",round_abs_list_average) # round_abs_list_average 0.15
    """ 


    # ----- 円形度画像の保存 -----#
    # print("save")
    # pil_image = Image.fromarray(np.uint8(label_image),mode = "P") # Numpy → pillow
    # pil_image.putpalette(palette)
    # pil_image.save("../mnt/kojima/pil_image.png")
    # -------------------------#

if __name__ == '__main__':

    args = sys.argv

    image     = Image.open("Color/" + args[1]+"/"+args[2]) # ("testdata/" + args[1]) 
    seg_image = Image.open("Palette.png")  # カラーパレット取得用
    print("image.size = ", image.size)

    base_width  = image.size[0]
    base_height = image.size[1]
    #image.save("1.jpg")
    # print(base_width)
    # print(base_height)
    
    # resize image
    image = image.resize((256, 256), Image.ANTIALIAS) #アンチエイリアスを掛けると写真縮小時のジャギジャギが軽減

    # delete alpha channel
    print("image.mode ==", image.mode) #image.mode == RGB
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # normalization
    image = np.asarray(image)
    # np.set_printoptions(threshold=np.inf) # printで省略せず表示する場合コメントアウトOFF
    # print(image.size)
    prepimg = image / 255.0

    # 1 Channel -> 3 Channels convert
    if prepimg.ndim < 3:
        prepimg = prepimg[:, :, np.newaxis]
        prepimg = np.insert(prepimg, 1, prepimg[:,:,0], axis=2)
        prepimg = np.insert(prepimg, 2, prepimg[:,:,0], axis=2)

    # Read .pb file
    with tf.gfile.FastGFile("../model/test.pb", "rb") as f:  ### モデルの指定
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(f.read())
        _ = tf.import_graph_def(graphdef, name="")
    sess = tf.Session()

    # Segmentation
    t1 = time.time()
    outputs = sess.run("output/BiasAdd:0", {"input:0":[prepimg]})   # 
    print("elapsedtime =", time.time() - t1)

    # Get a color palette
    palette = seg_image.getpalette()

    # Define index_void (len(DataSet.CATEGORY)-1)  分類するクラス数(背景,核,細胞質なので３クラス)
    index_void = 3

    # View
    output = outputs[0]
    res = np.argmax(output, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    #image = image.convert("RGB")
    image = image.resize((base_width, base_height))

    if '.jpg' in args[2]:
        output_name = args[2].replace(".jpg", "")
    elif '.jpeg' in args[2]:
        output_name = args[2].replace(".jpeg", "")
    elif '.png' in args[2]:
        output_name = args[2].replace(".png", "")

    image.save("Seg/Seg-" +args[1]+"/seg_"+ output_name + ".png")  # segmentation画像の保存
    calculate_circularity(image,output_name)
    print("complete No." + output_name)
    print()
"""
    # 入力画像とsegmentation画像の重ね合わせ処理
    im1 = Image.open("a/" + args[1])
    im2 = Image.open("Generated_seg_images/seg_" + output_name + ".png").resize(im1.size)
    
    mask = mask = Image.new("L", im1.size, 128)
    im = Image.composite(im1,im2,mask)
    im.save("./merge_images/merge_" + output_name + ".png")
"""
