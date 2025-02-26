from PIL import Image
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt


class Reporter:
    ROOT_DIR = "result"
    IMAGE_DIR = "image"
    LEARNING_DIR = "learning"
    INFO_DIR = "info"
    PARAMETER = "parameter.txt"
    IMAGE_PREFIX = "epoch_"
    IMAGE_EXTENSION = ".png"
    CIRCULARITY_DIR = "circularity"

    def __init__(self, result_dir=None, parser=None): # インスタンス生成時 パスを生成
        if result_dir is None:
            result_dir = Reporter.generate_dir_name()
        self._root_dir = self.ROOT_DIR # result
        self._result_dir = os.path.join(self._root_dir, result_dir) # result/20221011_2208(その時の時刻)
        self._image_dir = os.path.join(self._result_dir, self.IMAGE_DIR) # result/20221011_2208(その時の時刻)/image
        self._image_train_dir = os.path.join(self._image_dir, "train") # result/20221011_2208(その時の時刻)/image/train
        self._image_test_dir = os.path.join(self._image_dir, "test") # result/20221011_2208(その時の時刻)/image/test
        self._learning_dir = os.path.join(self._result_dir, self.LEARNING_DIR) # result/20221011_2208(その時の時刻)/learning
        self._info_dir = os.path.join(self._result_dir, self.INFO_DIR) # result/20221011_2208(その時の時刻)/info
        self._parameter = os.path.join(self._info_dir, self.PARAMETER) # result/20221011_2208(その時の時刻)/info/parameter.txt

        # ----------------- 円形度フォルダ
        # self._circularity_dir = os.path.join(self._result_dir, self.CIRCULARITY_DIR) # result/20221011_2208(その時の時刻)/circularity
        # -----------------
        self.create_dirs()

        self._matplot_manager = MatPlotManager(self._learning_dir)
        if parser is not None:
            self.save_params(self._parameter, parser)

    @staticmethod
    def generate_dir_name():
        return datetime.datetime.today().strftime("%Y%m%d_%H%M") # 実行した日付時間を返す

    def create_dirs(self): # ディレクトリの作成
        os.makedirs(self._root_dir, exist_ok=True)
        os.makedirs(self._result_dir)
        os.makedirs(self._image_dir)
        os.makedirs(self._image_train_dir)
        os.makedirs(self._image_test_dir)
        os.makedirs(self._learning_dir)
        os.makedirs(self._info_dir)
        # ----------------- 円形度フォルダ
        # os.makedirs(self._circularity_dir)
        # -----------------

    @staticmethod
    def save_params(filename, parser): # "info"フォルダにパラメーターを格納
        parameters = list()
        parameters.append("Number of epochs:" + str(parser.epoch))
        parameters.append("Batch size:" + str(parser.batchsize))
        parameters.append("Training rate:" + str(parser.trainrate))
        parameters.append("Augmentation:" + str(parser.augmentation))
        parameters.append("L2 regularization:" + str(parser.l2reg))
        output = "\n".join(parameters)

        with open(filename, mode='w') as f:
            f.write(output)

    def save_image(self, train, test, epoch):
        file_name = self.IMAGE_PREFIX + str(epoch) + self.IMAGE_EXTENSION
        train_filename = os.path.join(self._image_train_dir, file_name)
        test_filename = os.path.join(self._image_test_dir, file_name)
        train.save(train_filename) # /result/実行日時/train/epoch_0.png 的な形で保存
        test.save(test_filename)# /result/実行日時/test/epoch_0.png 的な形で保存

    def save_image_from_ndarray(self, train_set, test_set, palette, epoch, index_void=None):
        assert len(train_set) == len(test_set) == 3
        train_image = Reporter.get_imageset(train_set[0], train_set[1], train_set[2], palette, index_void)
        test_image = Reporter.get_imageset(test_set[0], test_set[1], test_set[2], palette, index_void)
        self.save_image(train_image, test_image, epoch)

    def create_figure(self, title, xylabels, labels, filename=None):
        return self._matplot_manager.add_figure(title, xylabels, labels, filename=filename)

    @staticmethod
    def concat_images(im1, im2, palette, mode):
        if mode == "P":
            assert palette is not None
            dst = Image.new("P", (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            dst.putpalette(palette)
        elif mode == "RGB":
            dst = Image.new("RGB", (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))

        return dst

    @staticmethod
    def cast_to_pil(ndarray, palette, index_void=None):
        assert len(ndarray.shape) == 3
        res = np.argmax(ndarray, axis=2)
        if index_void is not None:
            res = np.where(res == index_void, 0, res)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palette)
        return image

    @staticmethod
    def get_imageset(image_in_np, image_out_np, image_tc_np, palette, index_void=None):
        assert image_in_np.shape[:2] == image_out_np.shape[:2] == image_tc_np.shape[:2]

        # Image型:セグメンテーション結果画像, Image型:教師画像の生成 (numpy配列からImage型への変更)
        image_out, image_tc = Reporter.cast_to_pil(image_out_np, palette, index_void),\
                              Reporter.cast_to_pil(image_tc_np, palette, index_void)
        
        # Image型:セグメンテーション結果画像, Image型:教師画像 を結合
        image_concated = Reporter.concat_images(image_out, image_tc, palette, "P").convert("RGB")

        # Image型:オリジナル画像の生成
        image_in_pil = Image.fromarray(np.uint8(image_in_np * 255), mode="RGB")
        
        # Image型:オリジナル画像, Image型:セグメンテーション結果画像, Image型:教師画像 を結合
        image_result = Reporter.concat_images(image_in_pil, image_concated, None, "RGB")
        return image_result
    

    # ------------------------
    # circularity ディレクトリに学習過程の画像を全て保存
    # ------------------------
    def save_image_from_ndarray2(self, train_set, palette, epoch, batch_number, index_void=None):
        # assert len(train_set) == batch_size
        # print("batch",batch_number)
        # print("aaaaaa",train_set)
        for i in range(len(train_set)):
            train_image = Reporter.get_imageset2(train_set[i], palette, index_void)
            print("train_image", train_image)
            # train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
            # test_set = [test.images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
            self.save_image2(train_image, batch_number, epoch, i)
    
    @staticmethod
    def get_imageset2(image_out_np, palette, index_void=None):
        # assert image_in_np.shape[:2] == image_out_np.shape[:2] == image_tc_np.shape[:2]
        
        # Image型:セグメンテーション結果画像の生成 (numpy配列からImage型への変更)
        image_result= Reporter.cast_to_pil2(image_out_np, palette, index_void)
        
        return image_result

    @staticmethod
    def cast_to_pil2(ndarray, palette, index_void=None):
        assert len(ndarray.shape) == 3
        res = np.argmax(ndarray, axis=2)
        if index_void is not None:
            res = np.where(res == index_void, 0, res)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palette)
        return image
      
    def save_image2(self, train, batch_number, epoch, i):
        file_name = self.IMAGE_PREFIX + str(epoch) +"_"+ str("batch_num")+"_"+ str(batch_number)+"_"+ str(i)+ self.IMAGE_EXTENSION
        train_filename = os.path.join(self._circularity_dir, file_name)
        train.save(train_filename) # /result/実行日時/train/epoch_0_.png 的な形で保存
    # ------------------------


class MatPlotManager:
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._figures = {}

    def add_figure(self, title, xylabels, labels, filename=None):
        assert not(title in self._figures.keys()), "This title already exists."
        self._figures[title] = MatPlot(title, xylabels, labels, self._root_dir, filename=filename)
        return self._figures[title]

    def get_figure(self, title):
        return self._figures[title]


class MatPlot:
    EXTENSION = ".png"

    def __init__(self, title, xylabels, labels, root_dir, filename=None):
        assert len(labels) > 0 and len(xylabels) == 2
        if filename is None:
            self._filename = title
        else:
            self._filename = filename
        self._title = title
        self._xlabel, self._ylabel = xylabels[0], xylabels[1]
        self._labels = labels
        self._root_dir = root_dir
        self._series = np.zeros((len(labels), 0))

    def add(self, series, is_update=False):
        series = np.asarray(series).reshape((len(series), 1))
        assert series.shape[0] == self._series.shape[0], "series must have same length."
        self._series = np.concatenate([self._series, series], axis=1)
        if is_update:
            self.save()

    def save(self):
        plt.cla()
        for s, l in zip(self._series, self._labels):
            plt.plot(s, label=l)
        plt.legend()
        plt.grid()
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title(self._title)
        plt.savefig(os.path.join(self._root_dir, self._filename+self.EXTENSION))


if __name__ == "__main__":
    pass
