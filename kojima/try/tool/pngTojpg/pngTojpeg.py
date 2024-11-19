#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conda install pillow
"""
# 使用方法
○ディレクトリ構成
┃
┣ pngTojpeg
┃   ┣ pngTojpeg.py
┃   ┣ 画像フォルダ

○使用方法
①「input_path」にて「画像フォルダ」のパスを指定
②実行「python3 pngTojpeg.py」

○注意
--- png→jpgの場合 ---
img = img.convert('RGB')
img.save(save_filepath, "JPEG", quality=95)

--- jpg→pngの場合 ---
img = img.convert('RGBA')
img.save(save_filepath, "PNG", quality=95)
"""
import os, glob
from PIL import Image

def main():
    filepath_list = glob.glob(input_path + '/*.png') # .pngファイルをリストで取得する
    for filepath in filepath_list:
        basename  = os.path.basename(filepath) # ファイルパスからファイル名を取得
        save_filepath = out_path + '/' + basename [:-4] + '.jpg' # 保存ファイルパスを作成
        img = Image.open(filepath)
        img = img.convert('RGB') # RGBA(png)→RGB(jpg)へ変換
        img.save(save_filepath, "JPEG", quality=95)
        print(filepath, '->', save_filepath)
        if flag_delete_original_files:
            os.remove(filepath)
            print('delete', filepath)

if __name__ == '__main__':
    input_path = './bad_167' # オリジナルpngファイルがあるフォルダを指定
    out_path = input_path # 変換先のフォルダを指定
    flag_delete_original_files = True # 元ファイルを削除する場合は、True指定

    main()