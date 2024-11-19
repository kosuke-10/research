#! python3
# check09_3.py  -  フォルダ中で、連番ファイルの抜けをみつけ、連番が飛んでいる箇所を見つける。
#                  番号の飛びを埋めるように、後に続くファイルの名前を変更する。
# Usage
# py check09_3.py <path> <pre>

import os, sys, re, shutil

#ファイル名を取得
dir_file = sys.argv[1]
# pre = sys.argv[2]
file_dict = {}

file_list = os.listdir(dir_file)

for i in range(0, len(file_list)):
    file_dict[file_list[i]] = "{:03}.jpg".format(i+1)

#index的にはあるはずなのに、ファイル名として存在しないファイルを探す
def find_missing_file():
    for fname in file_dict.values():
        regex = re.compile(fname)
        mo = regex.search(", ".join(file_dict.keys()))
        if not mo:
            print(fname)


#元のファイル名から、連番の番号にする
def change_fname(change = False):
    if change == True:
        for k, v in file_dict.items():
            if k != v:
                print(k,v)
                shutil.move(os.path.join(dir_file, k), os.path.join(dir_file, v))
        
find_missing_file()

print("Do you want to change file name?:  y or [n]")
option = str(input())
if option == "y":
    change = True
else:
    change = False
change_fname(change)