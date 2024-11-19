# python concat.py 1
import sys
from PIL import Image

args = sys.argv

img1 = Image.open("Delete/OriginalImages/good_135/"+args[1] + ".png")
img2 = Image.open("Delete/SegmentationImages_handmade/good_135/" + args[1] + ".png")

# 水平に連結
def concat_side(img1, img2):
    concat_img = Image.new("RGB", (img1.width + img2.width, img1.height))
    concat_img.paste(img1, (0, 0))
    concat_img.paste(img2, (img1.width, 0))
    return concat_img

#垂直に連結 
def concat_vertical(img1, img2):
    concat_img = Image.new("RGB", (img1.width, img1.height + img2.height))
    concat_img.paste(img1, (0, 0))
    concat_img.paste(img2, (0, img1.height))
    return concat_img

concat_side(img1, img2).save("Delete/ConcatImages/good_135/" + args[1] + ".png")
# concat_vertical(img1, img2).save("Delete/ConcatImages/bad_167/" + args[1] + ".png")

print("Concat"+args[1]+"終了")
