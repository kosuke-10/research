#!/bin/bash

# シェルスクリプトを実行する回数を指定
num_images="$1"

# 各指標（Precision、Recall、Accuracy、F-measure）の合計を保存する変数
total_precision=0
total_recall=0
total_accuracy=0
total_f_measure=0

# 各スクリプトを指定回数実行し、各指標の合計を計算
for i in $(seq "$num_images"); do
  image_filename="$i.png"
  
  # precision.pyを実行し、結果を一時的な変数に保存
  precision_value=$(python3 precision.py "$image_filename" | awk -F ":" '{print $2}' | tr -d ' ')
  total_precision=$(awk "BEGIN{print $total_precision + $precision_value}")
  
  # recall.pyを実行し、結果を一時的な変数に保存
  recall_value=$(python3 recall.py "$image_filename" | awk -F ":" '{print $2}' | tr -d ' ')
  total_recall=$(awk "BEGIN{print $total_recall + $recall_value}")
  
  # accuracy.pyを実行し、結果を一時的な変数に保存
  accuracy_value=$(python3 accuracy.py "$image_filename" | awk -F ":" '{print $2}' | tr -d ' ')
  total_accuracy=$(awk "BEGIN{print $total_accuracy + $accuracy_value}")
  
  # f_measure.pyを実行し、結果を一時的な変数に保存
  f_measure_value=$(python3 f-measure.py "$image_filename" | awk -F ":" '{print $2}' | tr -d ' ')
  total_f_measure=$(awk "BEGIN{print $total_f_measure + $f_measure_value}")
done

# 各指標の平均を計算
average_precision=$(awk "BEGIN{print $total_precision / $num_images}")
average_recall=$(awk "BEGIN{print $total_recall / $num_images}")
average_accuracy=$(awk "BEGIN{print $total_accuracy / $num_images}")
average_f_measure=$(awk "BEGIN{print $total_f_measure / $num_images}")

# 結果を表示
echo "平均Precision: $average_precision"
echo "平均Recall: $average_recall"
echo "平均Accuracy: $average_accuracy"
echo "平均F-measure: $average_f_measure"
