import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from util import model_infer as model

#######################################################################################
### $ python3 main_infer.py　　#checkpointファイルのスリム化
# 
# できるファイル
# checkpoint
# deployfinal.ckpt.meta 
# deployfinal.ckpt.index
# data-00000-of-00001#
# semanticsegmentation_person_.pbtxt
#######################################################################################

def main():

    graph = tf.Graph()
    with graph.as_default():

        model_unet = model.UNet(l2_reg=0.0001).model

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("in=", model_unet.inputs.name)
        print("on=", model_unet.outputs.name)

        saver.restore(sess, './model/deploy.ckpt')
        saver.save(sess, './model/deployfinal.ckpt')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, './model', 'semanticsegmentation_person.pbtxt', as_text=True)

if __name__ == '__main__':
    main()
    print("軽量化完了")
