import tensorflow.compat.v1 as tf

def load_graph(model_file):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    return graph

model_file = '../model/test.pb'  # .pb ファイルのパスを指定してください
graph = load_graph(model_file)

for op in graph.get_operations():
    print(op.name)
    
    