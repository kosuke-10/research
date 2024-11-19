# ----------------
# eval.sh
# 評価指標値の算出
# N/C比の算出
# ----------------
echo ""
echo "Evaluation_Start"
for i in `seq $1`
do
echo "-- No."$i" --"
python3 accuracy.py $i.png  # 正解率
python3 recall.py $i.png    # 再現率
python3 precision.py $i.png # 適合率
python3 f-measure.py $i.png # F値
python3 find_nc_rate.py $i.png # N/C比
done
echo "Evaluation_End"
echo ""
