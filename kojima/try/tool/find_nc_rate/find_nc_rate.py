# python find_nc_rate.py 1.png
import sys
import io
from PIL import Image
import numpy  as np 

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

args = sys.argv

img = Image.open(args[1]).convert("RGB")
img_np = np.array(img)

nuclear = 0
cytoplasm = 0
background = 0
color_Palete = np.array([[[128, 0, 0],
                          [0, 128, 0],
                          [0, 0, 0]]])

for i in range(img_np.shape[0]):
    for j in range(img_np.shape[1]):
        if (img_np[i][j] == color_Palete[0][0]).all(): # nuclear:細胞核 赤色
            nuclear += 1
        elif (img_np[i][j] == color_Palete[0][1]).all(): # cytoplasm:細胞質 緑色
            cytoplasm += 1
        elif (img_np[i][j] == color_Palete[0][2]).all(): # background:背景 黒色
            background += 1 

# for i in range(img_np.shape[0]):
#     for j in range(img_np.shape[1]):
#         if img_np[i][j] == 2:
#             nuclear += 1
#         elif img_np[i][j] == 1:
#             cytoplasm += 1
#         elif img_np[i][j] == 0:
#             background += 1

# np.set_printoptions(threshold=np.inf) # printで省略せず表示する場合コメントアウトOFF
# print("check", color_Palete[0][0])
# print("check", img_np[25][25])
# print("check", img_np.shape[2])
# print("")
print(args[1])
print("n:", nuclear)
print("c:", cytoplasm)
print("b:", background)
nc_rate = nuclear/(nuclear + cytoplasm)
print("n/c比 =", nc_rate)
