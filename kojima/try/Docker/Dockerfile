FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get upgrade -y
RUN apt-get -y install python3.7
RUN apt-get -y install python3-pip python-qt4

RUN python3.7 -m pip install --upgrade pip

#パッケージインストール中の対話的なプロンプトが無効,デフォルトの選択肢が自動的に選択
ENV DEBIAN_FRONTEND=noninteractive
#タイムゾーン情報を格納するための標準的なディレクトリをインストール
RUN apt-get update && apt-get install -y tzdata

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install tensorflow==2.4.0
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install numpy==1.16.0
RUN pip install opencv-python
RUN pip install scikit-image==0.18.3
RUN pip install pandas
RUN pip install openpyxl
RUN apt-get install -y libgl1-mesa-dev
RUN pip install natsort
RUN pip install seaborn
RUN ln -f /usr/bin/python3.7 /usr/bin/python3

# 日本語フォントをインストール
RUN apt-get update && apt-get install -y fonts-noto-cjk

#環境変数指定(GPU読み込みに必要)
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64
#ハードリンク作成
RUN ln /usr/local/cuda-11.1/lib64/libcusolver.so.11 /usr/local/cuda-11.1/lib64/libcusolver.so.10

