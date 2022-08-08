#!/usr/bin/env python
'''
Kin Hasegawa氏によるatomcam.pyをCMOSカメラで利用できるようにした。オプションの-url
を１にすることで，カメラからのframeを読み込んでいる。
スレッド版に対して，逐次計算するするバージョンで現在テスト中。
'''
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta, timezone
import time
import argparse
import numpy as np
import cv2
import pafy
from imutils.video import FileVideoStream
import telnetlib



# 行毎に標準出力のバッファをflushする。
sys.stdout.reconfigure(line_buffering=True)


def composite(list_images):
    """画像リストの合成(単純スタッキング)
    Args:
      list_images: 画像データのリスト
    Returns:
      合成された画像
    """
    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)

    return output


def brightest(img_list):
    """比較明合成処理
    Args:
      img_list: 画像データのリスト
    Returns:
      比較明合成された画像
    """
    output = img_list[0]

    for img in img_list[1:]:
        output = np.where(output > img, output, img)

    return output


def diff(img_list):
    """画像リストから差分画像のリストを作成する。
    Args:
      img_list: 画像データのリスト
      mask: マスク画像(2値画像)
    Returns:
      差分画像のリスト
    """
    diff_list = []
    for img1, img2 in zip(img_list[:-2], img_list[1:]):
        # img1 = cv2.bitwise_or(img1)
        # img2 = cv2.bitwise_or(img2)
        diff_list.append(cv2.subtract(img1, img2))

    return diff_list


def detect(img, min_length):
    """画像上の線状のパターンを流星として検出する。
    Args:
      img: 検出対象となる画像
      min_length: HoughLinesPで検出する最短長(ピクセル)
    Returns:
      検出結果
    """
    blur_size = (5, 5)
    blur = cv2.GaussianBlur(img, blur_size, 0)
    canny = cv2.Canny(blur, 100, 200, 3)

    # The Hough-transform algo:
    return cv2.HoughLinesP(canny, 1, np.pi/180, 25, minLineLength=min_length, maxLineGap=5)


class AtomCam:
    def __init__(self, video_url, output=None, end_time="0600",
                 clock=False, mask=None, minLineLength=30):
        self._running = False
        # video device url or movie file path
        self.capture = None
        self.source = None

        self.url = video_url

        self.connect()
        self.FPS = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.HEIGHT = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # 出力先ディレクトリ
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path('.')
        self.output_dir = output_dir

        # MP4ファイル再生の場合を区別する。
        self.mp4 = False

        # 終了時刻を設定する。
        now = datetime.now()
        t = datetime.strptime(end_time, "%H%M")
        self.end_time = datetime(now.year, now.month, now.day, t.hour, t.minute)
        if now > self.end_time:
            self.end_time = self.end_time + timedelta(hours=24)

        print("# scheduled end_time = ", self.end_time)
        self.now = now

        self.min_length = minLineLength

    def __del__(self):
        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        print("# {} stop".format(obs_time))

        self.capture.release()
        cv2.destroyAllWindows()

    def connect(self):
        if self.capture:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.url)
        if self.url == 1:  # = USB Cmaera
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600) # Neptune-C 3096x2078
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400) # Mars-C 1944x1096
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, -6)
            self.capture.set(cv2.CAP_PROP_GAIN,80)
            self.capture.set(cv2.CAP_PROP_GAMMA,1)


    def stop(self):
        # thread を止める
        self._running = False


    def save_movie(self, img_list, pathname):
        """
        画像リストから動画を作成する。
        Args:
          imt_list: 画像のリスト
          pathname: 出力ファイル名
        """
        size = (self.WIDTH, self.HEIGHT)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        video = cv2.VideoWriter(pathname, fourcc, self.FPS, size)
        for img in img_list:
            video.write(img)

        video.release()

    def streaming(self, exposure, no_window):
        """
        ストリーミング再生
        Args:
          exposure: 比較明合成する時間(sec)
          no_window: True .. 画面表示しない
        Returns:
          0: 終了
          1: 異常終了
        """
        num_frames = int(self.FPS * exposure)
        composite_img = None

        while(True):
            # 現在時刻を取得
            now = datetime.now()
            obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)

            key = None
            img_list = []
            for n in range(num_frames):
                try:
                    ret, frame = self.capture.read()
                    frame = cv2.flip(frame,0)  # 上下反転（PlayeroneCamera)
                except Exception as e:
                    print(e, file=sys.stderr)
                    continue

                key = chr(cv2.waitKey(1) & 0xFF)
                if key == 'q':
                    return 0

                if not ret:
                    break

                if key == 's' and composite_img:
                    # 直前のコンポジット画像があれば保存する(未実装)。
                    # print(key)
                    pass

                img_list.append(frame)

            number = len(img_list)

            if number > 2:
                # 差分間で比較明合成を取るために最低3フレームが必要。
                # 画像のコンポジット(単純スタック)
                #composite_img = self.composite(img_list)
                # 画像のコンポジット(比較明合成)
                composite_img = brightest(img_list)
                diff_img = brightest(diff(img_list))
                try:
                    if not no_window:
                        blue_img = cv2.medianBlur(composite_img, 3)
                        show_img = cv2.resize(blue_img, (1050, 700))
                        cv2.imshow('Detecter x {} frames '.format(number), show_img)
                        # cv2.imshow('ATOM Cam2 x {} frames '.format(number), composite_img)
                    if detect(diff_img, self.min_length) is not None:
                        print('{} A possible meteor was detected.'.format(obs_time))
                        filename = "{:04}{:02}{:02}{:02}{:02}{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
                        path_name = str(Path(self.output_dir, filename + ".jpg"))
                        cv2.imwrite(path_name, composite_img)
                        # 検出した動画を保存する。
                        movie_file = str(Path(self.output_dir, "movie-" + filename + ".mp4"))
                        self.save_movie(img_list, movie_file)
                except Exception as e:
                    print(e, file=sys.stderr)
            else:
                print('No data: communcation lost? or end of data', file=sys.stderr)
                return 1

            # 終了時刻を過ぎたなら終了。
            now = datetime.now()
            if now > self.end_time:
                print("# end of observation at ", now)
                return 0



    def save_movie(self, img_list, pathname):
        """
        画像リストから動画を作成する。
        Args:
          imt_list: 画像のリスト
          pathname: 出力ファイル名
        """
        size = (self.WIDTH, self.HEIGHT)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        video = cv2.VideoWriter(pathname, fourcc, self.FPS, size)
        for img in img_list:
            video.write(img)

        video.release()



def streaming(args):
    """
    RTSPストリーミング、及び動画ファイルからの流星の検出
    (スレッドなし版、いずれ削除する。)
    """
    if args.url:
        atom = AtomCam(args.url, args.output, args.to)
        if not atom.capture.isOpened():
            return

    now = datetime.now()
    obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    print("# {} start".format(obs_time))

    while True:
        sts = atom.streaming(args.exposure, args.no_window)
        if sts == 1:
            if Path(args.url).suffix == '.mp4':
                # MP4ファイルの場合は終了する。
                return

            # 異常終了した場合に再接続する
            time.sleep(5)
            print("# re-connectiong to ATOM Cam ....")
            atom = AtomCam(args.url, args.output, args.to)
        else:
            return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # ストリーミングモードのオプション
    parser.add_argument('-u', '--url', default=1, help='RTSPのURL、または動画(MP4)ファイル')
    parser.add_argument('-n', '--no_window', action='store_true', help='画面非表示')


    # 共通オプション
    parser.add_argument('-e', '--exposure', type=int, default=1, help='露出時間(second)')
    parser.add_argument('-o', '--output', default=None, help='検出画像の出力先ディレクトリ名')
    parser.add_argument('-t', '--to', default="0600", help='終了時刻(JST) "hhmm" 形式(ex. 0600)')

    parser.add_argument('--min_length', type=int, default=30, help="minLineLength of HoghLinesP")
    parser.add_argument('-s', '--suppress-warning', action='store_true', help='suppress warning messages')

    # threadモード
    parser.add_argument('--thread', default=True, action='store_true', help='スレッド版')
    parser.add_argument('--help', action='help', help='show this help message and exit')

    args = parser.parse_args()

        # ストリーミング/動画(MP4)の再生(旧バージョン)
    streaming(args)

