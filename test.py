import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm

from networks.recognition.models import ArcFaceModel
from networks.resolution import srgan, vdsr
from utils import get_embeddings, crop_face_from_id, load_yaml


def evaluate_image(model, image1, image2):
    embeds1 = get_embeddings(model, image1)
    embeds2 = get_embeddings(model, image2)

    diff = np.subtract(embeds1, embeds2)
    dist = np.sum(np.square(diff), axis=1)
    return dist


if __name__ == "__main__":
    import csv
    import random
    import os

    faces = glob("data/faces/*.jpg")
    ids = glob("data/ids/*.jpg")

    model = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    ckpt_path = tf.train.latest_checkpoint("weights/arc_res50_kface_finetune_9K-lr0.001-bs128-epochs50")
    model.load_weights(ckpt_path)

    vdsr = vdsr.vdsr((112, 112, 3), 64)
    vdsr.load_weights("weights/vdsr-bs64-ps112.hdf5")

    with open("emb_dist-kface-epochs50-onebyone-final.csv", "w", encoding='utf-8', newline="") as f:
        writer = csv.writer(f)
        rands = np.arange(30)

        for i, id_path in tqdm(enumerate(ids)):
            # print(f"Iter {i+1:2d}: {os.path.basename(id_path)}")
            id_image = cv2.imread(id_path)
            id_image = crop_face_from_id(id_image)

            sr_id_image = cv2.resize(id_image, (112, 112))
            sr_id_image = vdsr.predict(np.expand_dims(sr_id_image, axis=0))
            sr_id_image = sr_id_image[0]

            # dist_rows = []
            # for face_path in tqdm(faces):
            #     face_image = cv2.imread(face_path)
            #     face_image = crop_face_from_id(face_image)
            #
            #     distance = evaluate_image(model, id_image, face_image)
            #     distance = list(distance)[0]
            #     dist_rows.append(distance)
            #
            # writer.writerow(dist_rows)
            face_path = os.path.join("data/faces", os.path.basename(id_path))
            face_image = cv2.imread(face_path)
            face_image = crop_face_from_id(face_image)

            distance = evaluate_image(model, id_image, face_image)
            sr_distance = evaluate_image(model, sr_id_image, face_image)
            distance = list(distance)[0]
            sr_distance = list(sr_distance)[0]
            writer.writerow([os.path.basename(id_path), os.path.basename(face_path), distance, sr_distance])

            rand_num = i
            while rand_num == i:
                rand_num = random.choice(rands)

            face_path = faces[rand_num]
            face_image = cv2.imread(face_path)
            face_image = crop_face_from_id(face_image)

            distance = evaluate_image(model, id_image, face_image)
            sr_distance = evaluate_image(model, sr_id_image, face_image)
            distance = list(distance)[0]
            sr_distance = list(sr_distance)[0]
            writer.writerow([os.path.basename(id_path), os.path.basename(face_path), distance, sr_distance])



    # with open("test_arcface.csv", "w", encoding='utf-8') as f:
    #     for id_path in ids[:7]:
    #         try:
    #             id_image = cv2.imread(id_path)
    #             id_image = crop_face_from_id(id_image)
    #         except:
    #             # f.write(id_path + " cannot be read")
    #             print(id_path + " cannot be read")
    #             continue
    #         for face_path in faces:
    #             f.write(',')
    #             try:
    #                 face_image = cv2.imread(face_path)
    #                 face_image = crop_face_from_id(face_image)
    #             except:
    #                 # f.write(face_path + " cannot be read")
    #                 f.write("img_error")
    #                 print(face_path + " cannot be read")
    #                 continue
    #
    #             try:
    #                 distance = evaluate_image(model, id_image, face_image)
    #                 # f.write(id_path + " " + face_path + " " + str(distance))
    #                 value = list(distance)[0]
    #                 f.write(str(value))
    #                 print(id_path + " " + face_path + " " + str(distance))
    #             except:
    #                 # f.write(id_path + face_path + " error")
    #                 f.write("emb_error")
    #                 print(id_path + " " + face_path + " error")
    #                 continue
    #
    #         f.write('\n')
