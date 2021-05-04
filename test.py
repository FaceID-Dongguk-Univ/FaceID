import cv2
import numpy as np
import tensorflow as tf
from glob import glob

from networks.recognition.models import ArcFaceModel
from utils import get_embeddings, crop_face_from_id


def evaluate_image(model, image1, image2):
    embeds1 = get_embeddings(model, image1)
    embeds2 = get_embeddings(model, image2)

    diff = np.subtract(embeds1, embeds2)
    dist = np.sum(np.square(diff), axis=1)
    return dist


if __name__ == "__main__":
    model = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    ckpt_path = tf.train.latest_checkpoint("weights/arc_res50_kface_finetune-lr0.001-bs128-trainable2")
    model.load_weights(ckpt_path)

    faces = glob("faces/*.jpg")
    ids = glob("ids/*.jpg")

    for id_path in ids:
        id_image = cv2.imread(id_path)
        id_image = crop_face_from_id(id_image)
        for face_path in faces:
            face_image = cv2.imread(face_path)
            face_image = crop_face_from_id(face_image)

            distance = evaluate_image(model, id_image, face_image)
            print(id_path, face_path, distance)
