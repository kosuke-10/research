###################################################################
# $ python3 main.py --gpu --augmentation --batchsize 32 --epoch 50
###################################################################

import argparse
import random
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

def train(parser):

    # Load train and test datas
    train, test = load_dataset(train_rate=parser.trainrate)
    # 検証用のデータセット
    valid = train.perm(0, 30) # 30
    test = test.perm(0, 150) # 150

    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # Whether or not using a GPU
    gpu = parser.gpu # true or false が格納

    # Create a model
    model_unet = model.UNet(l2_reg=parser.l2reg).model

    # Set a loss function and an optimizer   
    # reduce_mean():平均値を返す 
    # labels:教師データ logits:最終的な推計値
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) 

    # Calculate accuracy
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

    s_time = time.time() # 学習開始時刻を格納 単位:秒数
    
    print()
    print("Training_Start")
    for epoch in range(epochs):
        for batch in train(batch_size=batch_size, augment=is_augment):
            # Expansion of batch data
            inputs = batch.images_original
            teacher = batch.images_segmented
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})

        # Evaluation
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test,  " Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)#"accuracy"をグラフに追加
            loss_fig.add([loss_train, loss_test], is_update=True)#"loss"をグラフに追加
            if epoch % export_fre == 0:  # default 3
                idx_train = random.randrange(150) #10
                idx_test = random.randrange(150) #100
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train.images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [test.images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
                test_set = [test.images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                                                 index_void=len(ld.DataSet.CATEGORY)-1)
        saver.save(sess, './model/deploy.ckpt')  # 学習パラメータの保存
        print("in=", model_unet.inputs.name)
        print("on=", model_unet.outputs.name)

    # Test the trained model
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
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
