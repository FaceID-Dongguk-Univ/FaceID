import numpy as np
import zipfile
import os
import shutil
import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob


def main():
    # --- wrapper unzip ---
    if not os.path.exists("Middle_Resolution"):
        os.makedirs("Middle_Resolution")

    zipfile.ZipFile("Middle_Resolution.zip").extractall("Middle_Resolution")

    # --- unzip ---
    zip_names = glob("Middle_Resolution/*.zip")

    class_names = []
    for z in zip_names:
        class_names.append(os.path.basename(z).split('.')[0])

    lux = ["L1", "L3", "L6"]
    emotion = ["E01", "E02", "E03"]
    angle = ["C5", "C6", "C7", "C8", "C9", "C10"]
    img_names = []
    txt_names = []

    for l in lux:
        for e in emotion:
            for c in angle:
                img_names.append(l + '/' + e + '/' + c + '.jpg')
                txt_names.append(l + '/' + e + '/' + c + '.txt')

    for z, c in tqdm(zip(zip_names, class_names)):
        if not os.path.exists("MR/" + c):
            os.makedirs("MR/" + c)

        for i, t in zip(img_names, txt_names):
            zipfile.ZipFile(z).extract("S001/" + i)
            zipfile.ZipFile(z).extract("S001/" + t)
        shutil.move("./S001", "MR/" + c)

        print(c + "done")

    # --- crop ---
    for j, c in enumerate(class_names):
        base = "kface/" + str(j)
        if not os.path.exists(base):
            os.makedirs(base)

        imgs = glob("MR/" + c + "/*/*/*/*.jpg")
        txts = glob("MR/" + c + "/*/*/*/*.txt")

        for i, (img, txt) in enumerate(zip(imgs, txts)):
            name = str(i)
            with open(txt, 'r') as f:
                bbox = f.read().split('\n')[7].split()
                bbox = list(map(int, bbox))
                (x, y, w, h) = bbox

                img = cv2.imread(img)
                img = img[y: y + h, x: x + w]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                Image.fromarray(img).save(os.path.join(base, str(j) + '_' + name) + '.jpg')
                Image.fromarray(img).save("kface_SR/" + str(j) + '_' + name + '.jpg')

    # --- remove directories ---
    shutil.rmtree("MR")
    # shutil.rmtree("Middle_Resolution")


if __name__ == "__main__":
    main()
