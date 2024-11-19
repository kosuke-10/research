from PIL import Image
import numpy as np
import glob
import os
from util import image_augmenter as ia


class Loader(object):
    def __init__(self, dir_original, dir_segmented, init_size=(256, 256), one_hot=True): # コンストラクタ(インスタンス生成時に実行=初期化)
        self._data = Loader.import_data(dir_original, dir_segmented, init_size, one_hot) # selfにはloaderが格納
        """
        self._data._images_original = images_original
        self._data._images_segmented = images_segmented
        self._data._image_palette = image_palette
        self._data._augmenter = augmenter
        """

    def get_all_dataset(self):
        return self._data

    def load_train_test(self, train_rate=0.85, shuffle=True, transpose_by_color=False):
        """
        `Load datasets splited into training set and test set.
        Args:
            train_rate (float): Training rate.
            shuffle (bool): If true, shuffle dataset.
            transpose_by_color (bool): If True, transpose images for chainer. [channel][width][height]
        Returns:
            Training Set (Dataset), Test Set (Dataset)
        """
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")
        if transpose_by_color:
            self._data.transpose_by_color()
        if shuffle:
            self._data.shuffle()

        """
        train_size : Training時に使用する画像の枚数を格納 # "self._data.images_original.shape[0]" の部分は "self._data.images_segmented" でも可
        data_size  : 使用する全画像枚数を格納
        train_set  : 「images_original」(inputs)と「images_segmented」(teacher)のTraining時に使用する画像枚数分格納
        test_set   : 「images_original」(inputs)と「images_segmented」(teacher)のEvaluation時に使用する画像枚数分格納
        """
        train_size = int(self._data.images_original.shape[0] * train_rate) # 922 = 水増し後の画像枚数 ×　train_rate
        data_size = int(len(self._data.images_original)) # 1085 = 水増し後の画像枚数
        train_set = self._data.perm(0, train_size)
        test_set = self._data.perm(train_size, data_size)

        """
        Q.
        print("train_size:",train_size) # 922
        print("data_size:",data_size) # 1085

        <train>
        print("train_set._images_original.shape",train_set._images_original.shape)
        print("train_set._images_segmented.shape",train_set._images_segmented.shape)
        
        <test>
        print("test_set._images_original.shape",test_set._images_original.shape)
        print("test_set._images_segmented.shape",test_set._images_segmented.shape)

        A.
        train_size: 922
        data_size: 1085

        <train>
        train_set._images_original.shape:train_set (922, 256, 256, 3)
        train_set._images_segmented.shape:train_set (922, 256, 256, 3)

        <test>
        test_set._images_original.shape:test_set (163, 256, 256, 4)
        test_set._images_segmented.shape:test_set (163, 256, 256, 4)
        """

        return train_set, test_set

    @staticmethod #@staticmethodにすることでインスタンス名を第１引数selfとして格納しなくて済む
    def import_data(dir_original, dir_segmented, init_size=None, one_hot=True):
        # Generate paths of images to load(読み込む画像のパスを生成)
        paths_original, paths_segmented = Loader.generate_paths(dir_original, dir_segmented)
        #print(paths_segmented) = ['../mnt/kojima/dataset/JPEGImagesOUT/0005_0004.jpg', '../mnt/kojima/dataset/JPEGImagesOUT/0010_0032.jpg'....]

        # Extract images to ndarray using paths(パスを用いて画像をndarrayに展開)4次元のnumpy型を格納
        images_original, images_segmented = Loader.extract_images(paths_original, paths_segmented, init_size, one_hot)

        # Get a color palette
        image_sample_palette = Image.open(paths_segmented[0])  # PILでopenすることでインデックス値をRGB値に変換することなく取得できる
        palette = image_sample_palette.getpalette()  # getpalette カラーパレットにアクセスする
        # リストの値はindex=0から順番に[R,G,B,R,G,B, ...] (256*3)の一次元配列

        return DataSet(images_original, images_segmented, palette,
                       augmenter=ia.ImageAugmenter(size=init_size, class_count=len(DataSet.CATEGORY)))

    @staticmethod
    def generate_paths(dir_original, dir_segmented): #(パスの生成)
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0: # 該当するファイル名のファイルが存在しなかったとき
            raise FileNotFoundError("Could not load images.")
        # map(function, iterable) functionを結果を返しながらiterableの全ての要素に適応するイテレータ(集合データ)を返す
        # 無名関数lambda 使い捨ての関数として使用
        # lamdbda 引数1, 引数2, …: 式
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))  # 拡張子の前までを取る os.sep = / ←linuxの場合
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames)) # 拡張子をjpgに変換

        return paths_original, paths_segmented

    @staticmethod
    def extract_images(paths_original, paths_segmented, init_size, one_hot): #(イメージ画像の展開)
        images_original, images_segmented = [], []

        # Load images from directory_path using generator(ディレクトリパスからジェネレータで画像を読み込む)
        print("Loading original images", end="", flush=True)
        for image in Loader.image_generator(paths_original, init_size, antialias=True):  # 入力画像のアルファチャンネルを削除し、リサイズ処理を行う(256^2の正方形処理)
            images_original.append(image)
            if len(images_original) % 200 == 0:  # 200枚読み込むごとに
                print(".", end="", flush=True)  # end="" 改行をなくす
        print(" Completed", flush=True)

        print("Loading segmented images", end="", flush=True)
        for image in Loader.image_generator(paths_segmented, init_size, normalization=False):
            images_segmented.append(image)
            if len(images_segmented) % 200 == 0:
                print(".", end="", flush=True)
        print("Completed")
        assert len(images_original) == len(images_segmented)  # assert 条件式がFalseのとき、AssertionErrorの例外を発生 水増し後の画像の枚数

        #-------- ここまでで画像をテンソルとして格納完了 ---------#

        # Cast to ndarray テンソルをnumpy型のリストへ変換
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)  # np.uint8

        # Change indices which correspond to "void" from 255  
        # クラスインデックスの書き換え
        # 分類したいクラス以外を黒で塗りつぶし　→　分類したいクラスに1,2,…とインデックス番号を振っていく
        images_segmented = np.where((images_segmented != 1) & (images_segmented != 2) & (images_segmented != 255), 0, images_segmented)  # personクラスと境界線以外のクラスインデックスを0に置き換え (分類したいクラス以外を塗りつぶし)
        #images_segmented = np.where(images_segmented == 15, 1, images_segmented)  # 15(personクラス)のインデックスを1にする
        #images_segmented = np.where(images_segmented == 8, 2, images_segmented)
        images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)  # void(255)をクラス数にする

        # One hot encoding using identity matrix.
        if one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)  # identity(n, dtype = float) n*nの単位行列を生成
            images_segmented = identity[images_segmented]
            print("Done")
        else:
            pass

        return images_original, images_segmented

    @staticmethod
    def cast_to_index(ndarray):
        return np.argmax(ndarray, axis=2)

    @staticmethod
    def cast_to_onehot(ndarray):
        identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
        return identity[ndarray]

    @staticmethod
    def image_generator(file_paths, init_size=None, normalization=True, antialias=False): #file_paths = paths_original or paths_segmented 全画像のパスの文字列リスト
        """
        `A generator which yields images deleted an alpha channel and resized.
        アルファチャネル削除、リサイズ(任意)処理を行った画像を返します
        Args:
            file_paths (list[string]): File paths you want load.
            init_size (tuple(int, int)): If having a value, images are resized by init_size.
            normalization (bool): If true, normalize images.
            antialias (bool): Antialias.
        Yields:
            image (ndarray[width][height][channel]): Processed image
        """
        for file_path in file_paths: # file_paths = ['../mnt/kojima/dataset/JPEGImagesOUT/0005_0004.jpg', '../mnt/kojima/dataset/JPEGImagesOUT/0010_0032.jpg'....]
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image(イメージ画像を開く)
                image = Image.open(file_path)
                # to square(画像の中心部を基準として正方形に画像を切り取り)
                image = Loader.crop_to_square(image)
                # resize by init_size
                if init_size is not None and init_size != image.size:  # init_sizeが指定されていれば入力画像をリサイズ
                    if antialias:
                        image = image.resize(init_size, Image.ANTIALIAS)  # ANTIALIAS 画像を縮小した際にギザギザになるのを軽減する
                    else:
                        image = image.resize(init_size)
                # delete alpha channel
                if image.mode == "RGBA":  # modeでImageモードを取得
                    image = image.convert("RGB")  # convertでモードの書き換え
                image = np.asarray(image)   # asarray 引数がndarrayのときコピーされる
                if normalization: # RGBの各値を225で割って正規化(0~1)
                    image = image / 255.0
                yield image  # yield　関数を一時的に実行停止させることができる機能を持つ  メモリの消費量を抑えることができる

    @staticmethod
    def crop_to_square(image):
        size = min(image.size)  # sizeは(幅、高さ)
        left, upper = (image.width - size) // 2, (image.height - size) // 2  # (left, upper)左上の座標
        right, bottom = (image.width + size) // 2, (image.height + size) // 2  # (right, bottom)右下の座標
        return image.crop((left, upper, right, bottom))  # 切り取り 画像の中心部を基準として正方形を切り取っている







class DataSet(object):
    CATEGORY = (
        "ground",
        "cell",
        "cytoplasm",
        "void"
    )

    def __init__(self, images_original, images_segmented, image_palette, augmenter=None):
        assert len(images_original) == len(images_segmented), "images and labels must have same length."
        self._images_original = images_original
        self._images_segmented = images_segmented
        self._image_palette = image_palette
        self._augmenter = augmenter

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def palette(self):
        return self._image_palette

    @property
    def length(self):
        return len(self._images_original)

    @staticmethod
    def length_category():
        return len(DataSet.CATEGORY)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images_original))

    def __add__(self, other):
        images_original = np.concatenate([self.images_original, other.images_original])
        images_segmented = np.concatenate([self.images_segmented, other.images_segmented])
        return DataSet(images_original, images_segmented, self._image_palette, self._augmenter)

    def shuffle(self):
        idx = np.arange(self._images_original.shape[0])
        np.random.shuffle(idx)
        self._images_original, self._images_segmented = self._images_original[idx], self._images_segmented[idx]

    def transpose_by_color(self):
        self._images_original = self._images_original.transpose(0, 3, 1, 2)
        self._images_segmented = self._images_segmented.transpose(0, 3, 1, 2)

    def perm(self, start, end):
        end = min(end, len(self._images_original)) # self._images_segmented でも良い
        return DataSet(self._images_original[start:end], self._images_segmented[start:end], self._image_palette,
                       self._augmenter)

    def __call__(self, batch_size=20, shuffle=True, augment=True):
        """
        `A generator which yields a batch. The batch is shuffled as default.
        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.
        Yields:
            batch (ndarray[][][]): A batch data.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        if shuffle:
            self.shuffle()

        for start in range(0, self.length, batch_size): # self.length = 922
            """
            print(list(range(0, 10, 3)))
                # [0, 3, 6, 9]
            """
            batch = self.perm(start, start+batch_size)
            if augment:
                assert self._augmenter is not None, "you have to set an augmenter."
                yield self._augmenter.augment_dataset(batch, method=[ia.ImageAugmenter.NONE, ia.ImageAugmenter.FLIP])
            else:
                yield batch







if __name__ == "__main__":
    dataset = "dataset"
    dataset_loader = Loader(dir_original="../mnt/kojima/" + dataset + "/JPEGImages",
                            dir_segmented="../mnt/kojima/" + dataset + "/SegmentationClass")
    train, test = dataset_loader.load_train_test()
    train.print_information()
    test.print_information()
