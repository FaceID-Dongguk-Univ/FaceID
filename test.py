import cv2
import numpy as np
import tensorflow as tf

from networks.recognition.models import ArcFaceModel
from utils import get_embeddings, crop_face_from_id


def evaluate_image(image1, image2):
    model = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    ckpt_path = tf.train.latest_checkpoint("weights/arc_res50_kface_finetune-lr0.001-bs128-trainable2")
    model.load_weights(ckpt_path)
    embeds1 = get_embeddings(model, image1)
    embeds2 = get_embeddings(model, image2)

    diff = np.subtract(embeds1, embeds2)
    dist = np.sum(np.square(diff), axis=1)
    return dist


if __name__ == "__main__":
    id_image = cv2.imread("resource/ush.jpeg")
    id_image = crop_face_from_id(id_image)

    face_image = cv2.imread("resource/ush_face.jpg")
    face_image = crop_face_from_id(face_image)

    distance = evaluate_image(id_image, face_image)
    print(distance)
