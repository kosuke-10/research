from matplotlib import font_manager

fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# 利用可能なフォントのリストを表示する
print(fonts)

# 日本語を含む可能性のあるフォントの名前を表示する
for font in fonts:
    if 'japan' in font.lower() or 'cjk' in font.lower() or 'gothic' in font.lower():
        print(font)
