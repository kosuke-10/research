# ----------------
# docker.sh
# dockerimageの作成
# dockerコンテナの作成
# docker.sh "GPU番号"
# ----------------

docker build -t yoshida/u-net:11.1.1-cudnn8-devel-ubuntu18.04 .
docker run --gpus "device=$1" -it -v /home/yoshida/kojima/try/U-net:/U-net -v /home/yoshida/kojima/try/mnt:/mnt --name yoshida_u-net$2 yoshida/u-net:11.1.1-cudnn8-devel-ubuntu18.04
