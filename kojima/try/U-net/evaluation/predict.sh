for i in `seq $1`
do
python3 predict.py UC004 $i.png
done