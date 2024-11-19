# ----------------
# make_pb.sh
# pbファイルの生成
# ----------------

python3 freeze_graph.py --input_graph=model/semanticsegmentation_person.pbtxt --input_checkpoint=model/deployfinal.ckpt --output_graph=model/test.pb --output_node_names=output/BiasAdd --input_binary=False