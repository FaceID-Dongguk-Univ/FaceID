import numpy as np
import zipfile
import os
import shutil
import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob


def wrapper_unzip():
    if not os.path.exists("Middle_Resolution"):
        os.makedirs("Middle_Resolution")

    zipfile.ZipFile("Middle_Resolution.zip").extractall("Middle_Resolution")


def unzip():
    zip_names = glob("Middle_Resolution/*.zip")

    class_names = []
    for z in zip_names:
        class_names.append(os.path.basename(z).split('.')[0])

    lux = ["L1", "L3"]
    emotion = ["E01", "E02", "E03"]
    angle = ["C6", "C7", "C8", "C9"]
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
        shutil.move("S001", "MR/" + c)

        # if not os.path.exists("kface_SR"):
        #     os.makedirs("kface_SR")
        # if not os.path.exists("kface_SR_val"):
        #     os.makedirs("kface_SR_val")

    # --- crop ---
    for j, c in enumerate(class_names):

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

                if j >= 390:
                    base_val = "kface_val/" + str(j - 390)
                    if not os.path.exists(base_val):
                        os.makedirs(base_val)
                    Image.fromarray(img).save(os.path.join(base_val, str(j-390) + '_' + name) + '.jpg')
                    # Image.fromarray(img).save("kface_SR_val/" + str(j-390) + '_' + name + '.jpg')
                else:
                    base = "kface/" + str(j)

                    if not os.path.exists(base):
                        os.makedirs(base)
                    Image.fromarray(img).save(os.path.join(base, str(j) + '_' + name) + '.jpg')
                    # Image.fromarray(img).save("kface_SR/" + str(j) + '_' + name + '.jpg')


def make_pair_numpy():
    total_list = glob("kface_val/*/*.jpg")

    ref_list = []
    for i in range(10):
        img_name = np.random.randint(24)
        ref_list.append(os.path.join("kface_val", f"{i}", f"{i}_{img_name}.jpg"))

    query_list = total_list.copy()
    for i in ref_list:
        query_list.remove(i)

    ref_imgs = []
    query_imgs = []
    is_same = []
    classes = np.arange(10)

    for class_id, ref_img in enumerate(ref_list):
        base = "kface_val"
        for i in range(40):
            ref_imgs.append(np.array(Image.open(ref_img).resize((112, 112)), dtype=np.uint8))

        for i in range(20):
            same_num = str(np.random.randint(24))
            same_img = os.path.join(base, f'{class_id}', f'{class_id}_{same_num}.jpg')
            query_imgs.append(np.array(Image.open(same_img).resize((112, 112)), dtype=np.uint8))
            is_same.append(1)

            dif_num = str(np.random.randint(24))
            another_class_id = np.random.choice(np.delete(classes, class_id))
            dif_img = os.path.join(base, f'{another_class_id}', f'{another_class_id}_{dif_num}.jpg')
            query_imgs.append(np.array(Image.open(dif_img).resize((112, 112)), dtype=np.uint8))
            is_same.append(0)

    ref_imgs = np.array(ref_imgs, dtype=np.uint8)
    query_imgs = np.array(query_imgs, dtype=np.uint8)
    is_same = np.array(is_same, dtype=np.uint8)

    if not os.path.exists('kface_val_npy'):
        os.makedirs('kface_val_npy')

    np.save('kface_val_npy/references.npy', ref_imgs)
    np.save('kface_val_npy/queries.npy', query_imgs)
    np.save('kface_val_npy/is_same.npy', is_same)


def main():
    wrapper_unzip()

    unzip()

    shutil.rmtree("MR")
    shutil.rmtree("Middle_Resolution")

    make_pair_numpy()


if __name__ == "__main__":
    main()
