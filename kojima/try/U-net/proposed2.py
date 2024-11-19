###################################################################
# $ python3 main.py --gpu --augmentation --batchsize 32 --epoch 50
###################################################################

import argparse
from pickle import FALSE
import random
from PIL import Image
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import skimage
from skimage.measure import label, regionprops_table
from skimage.draw import random_shapes
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import time

from util import loader as ld
from util import model as model
from util import repoter as rp

dataset = "dataset"

def load_dataset(train_rate):
    loader = ld.Loader(dir_original="../mnt/" + dataset + "/JPEGImagesOUT", 
                       dir_segmented="../mnt/" + dataset + "/SegmentationClassOUT") 
    return loader.load_train_test(train_rate=train_rate, shuffle=False)

def calculate_circularity(epoch, train_set, palette, index_void=None):
    # 画像の１枚１枚の円形度を格納するための空のリストを作成
    circularity_list = []

    for i in range(len(train_set)):
            train_image = get_imageset(train_set[i], palette, index_void) # Numpy → pillow # train_image.saveで画像を1枚ずつ保存可能
            # train_image.save("../mnt/train_image.png") # 画像を1枚ずつ保存したい場合はアンコメント

            train_image_array = np.asarray(train_image) # pillow → Numpy
            copy_image = np.copy(train_image_array) # WRITEABLE : False → True
            Average_value_of_circularity_of_one_image = Average_value(copy_image, palette, epoch)

            circularity_list.append(Average_value_of_circularity_of_one_image) # 画像１枚の円形度の平均値を格納

    return st.mean(circularity_list)
 
def get_imageset(image_out_np, palette, index_void=None):
        # Image型:セグメンテーション結果画像の生成 (numpy配列からImage型への変更)
        image_result= cast_to_pil(image_out_np, palette, index_void)
        
        return image_result

def cast_to_pil(ndarray, palette, index_void=None):
        assert len(ndarray.shape) == 3 # リストの形が3次元
        res = np.argmax(ndarray, axis=2)

        if index_void is not None:
            res = np.where(res == index_void, 0, res) # res == index_voidが真ならre0, 偽ならresに 
        image = Image.fromarray(np.uint8(res), mode="P") # Numpy → pillow
        image.putpalette(palette)
        return image

# ----------
# Average_value関数:1枚の画像の円形度の平均を求める関数
# 返り値:int型
# ----------
def Average_value(train_image_array, palette, epoch): # 1枚の画像の円形度の平均を求める関数  
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

    #ラベリング領域の情報をhtml形式で保存したい場合はアンコメント
    #df.to_html('rs_label_result.html')

    # #円形度を画像上に表示
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(label_image_f,cmap="turbo",alpha=0.5)
    for n in df.index.values:
        ax.text(df["centroid-1"][n], df["centroid-0"][n],np.round(df["circularity"][n],2)) # ax.text(x,y,"文字列",size)
        circularity_list.append(np.round(df["circularity"][n],2))
    ax.set_title("Circularity")       
    # plt.savefig("random_shapes_circularity.png") # 画像を保存する場合アンコメント
    plt.close()
    if len(circularity_list) == 0:
        circularity_list.append(0)

    list_average = st.mean(circularity_list)
    round_abs_list_average = round(abs(list_average-1),2) # 円形度を-1し正規化

    return round_abs_list_average

def train(parser):

    # Load train and test datas
    train, test = load_dataset(train_rate=parser.trainrate) # train_rate=0.85 水増し後のデータセットを0.85(train):0.15(test)の比率に分配

    # 検証用のデータセットを全データセットから切り取る Evaluation時に使われるデータセットになる 学習時には使用しない
    valid = train.perm(0, 30) 
    test = test.perm(0, 150) 

    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # Whether or not using a GPU
    gpu = parser.gpu # true or false が格納

    # Create a model
    model_unet = model.UNet(l2_reg=parser.l2reg).model

    # Set a loss function and an optimizer   
    # reduce_mean():平均値を返す デフォルトでaxis = Noneであるため単一のテンソルが帰ってくる = 1つの値が帰ってくる
    # labels:教師データ logits:最終的な推計値
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    
    circularity = tf.compat.v1.placeholder(tf.float32)
    
    """
    tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,logits=model_unet.outputs)
    ↑ 返り値:3次元配列 ←　32枚分(最後のbatchは異なるが)の画素ごとのセグ結果画像と教師画像の誤差が格納
    """

    # 損失関数を引数として渡すことで損失関数の値が小さくなるように重みパラメータを調整
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        train_step2 = tf.train.AdamOptimizer(1e-3).minimize(tf.reduce_mean([cross_entropy, circularity]))
    
    # Calculate accuracy
    # テストデータの正解画像とセグメンテーションした画像を1画素づつ比較しそれぞれのクラスの再現率の平均を取る
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize session
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train the model
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation
    export_fre = parser.export_fre
    train_dict = {model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,
                  model_unet.is_training: False}
    test_dict = {model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,
                 model_unet.is_training: False}

    flag = 0

    s_time = time.time() # 学習開始時刻を格納 単位:秒数

    print()
    print("Training_Start")
    for epoch in range(epochs):
        print("epoch:" + str(epoch)+ " start")
        batch_number = 0
        for batch in train(batch_size=batch_size, augment=is_augment):
            print("batch:" + str(batch_number)+ " start")

            inputs = batch.images_original
            teacher = batch.images_segmented
            
            if flag == 0:
                # Training (重みパラメータの調整含む)
                sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,model_unet.is_training: True})

            if flag == 1:
                outputs_images = sess.run(model_unet.outputs, feed_dict={model_unet.inputs: inputs, model_unet.is_training: False})
                passed_circularity = calculate_circularity(epoch, outputs_images, train.palette, index_void=len(ld.DataSet.CATEGORY)-1) # 32枚の画像の円形度の平均を求める格納

                # Training (重みパラメータの調整含む)
                sess.run(train_step2, feed_dict={circularity: passed_circularity, model_unet.inputs: inputs, model_unet.teacher: teacher, model_unet.is_training: True})

            # --------------
            # 画像データの保存を保存したい場合以下をアンコメント
            # --------------
            # outputs_images = sess.run(model_unet.outputs, feed_dict={model_unet.inputs: inputs, model_unet.is_training: False})
            # reporter.save_image_from_ndarray2(outputs_images, train.palette, epoch, batch_number, index_void=len(ld.DataSet.CATEGORY)-1)
            # --------------
            print("batch:" + str(batch_number)+ " end")
            batch_number += 1

        # Evaluation (1エポック終了ごとに評価)
        if epoch % 1 == 0:
            if flag == 0:
                loss_train = sess.run(cross_entropy, feed_dict=train_dict)
                loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            if flag == 1:
                loss_train = sess.run(tf.reduce_mean([cross_entropy, circularity]), feed_dict={circularity: passed_circularity, model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,model_unet.is_training: False})
                loss_test = sess.run(tf.reduce_mean([cross_entropy, circularity]), feed_dict={circularity: passed_circularity, model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,model_unet.is_training: False})
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test,  " Accuracy:", accuracy_test)

            if flag == 0:
                if loss_test < 0.1:
                    flag = 1
                    # train_step2 = train_step
                    print("円形度算出開始")

            # グラフに追記
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)#"accuracy"をグラフに追加
            loss_fig.add([loss_train, loss_test], is_update=True)#"loss"をグラフに追加

            # 3epockごとに学習状況を画像出力保存
            if epoch % export_fre == 0:  # default 3
                idx_train = random.randrange(150) #0 ~ 150までのランダム値を代入 出力する画像の番号になる
                idx_test = random.randrange(150)
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train.images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [test.images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]] # 3種の画像をnumpy型で格納 3次元の配列
                test_set = [test.images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]

                #３種の画像を繋げて出力
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                                                 index_void=len(ld.DataSet.CATEGORY)-1)
                
        saver.save(sess, './model/deploy.ckpt')  # 学習パラメータの保存
        print("in=", model_unet.inputs.name)
        print("on=", model_unet.outputs.name)

        print("epoch:" + str(epoch)+ " end")

    # Test the trained model
    if flag == 0:
        loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    if flag == 1:
        loss_test = sess.run(tf.reduce_mean([cross_entropy, circularity]), feed_dict={circularity: passed_circularity, model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,model_unet.is_training: False})
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
    print("elapsed_time:", (time.time() - s_time)/60) #経過時間
    print()
    print("Training_End")

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    # 受け取る引数を追加していく
    # -- はオプション引数(指定しなくてもいい引数)
    # '-a', '--arg4'は省略形
    # default :デフォルト値を指定  type :型を指定  action :フラグとして使用
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.85, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('-ef', '--export_fre', type=int, default=3, help='Export frequency')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
