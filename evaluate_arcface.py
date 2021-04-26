"""
An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition
https://arxiv.org/abs/1801.07698

This ArcFace code is based on peteryuX's arcface-tf2.
https://github.com/peteryuX/arcface-tf2
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import l2_norm
from networks.recognition.models import ArcFaceModel


def get_embedding(model, x):
    if len(x.shape) == 3:
        x = np.expand_dims(x, 0)
    embeds = model.predict(x)
    embeds = l2_norm(embeds)
    return embeds


def calculate_accuracy(threshold, dist, actual_is_same):
    predict_is_same = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_is_same, actual_is_same))
    fp = np.sum(np.logical_and(predict_is_same, np.logical_not(actual_is_same)))
    tn = np.sum(np.logical_and(np.logical_not(predict_is_same), np.logical_not(actual_is_same)))
    fn = np.sum(np.logical_and(np.logical_not(predict_is_same), actual_is_same))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_is_same):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    tprs = np.zeros((len(thresholds)))
    fprs = np.zeros((len(thresholds)))
    accuracy = np.zeros((len(thresholds)))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), axis=1)
    # diff = tf.subtract(embeddings1, embeddings2)
    # dist = tf.reduce_sum(tf.square(diff), axis=1)

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], accuracy[threshold_idx] = calculate_accuracy(threshold, dist, actual_is_same)
    best_threshold_idx = np.argmax(accuracy)
    best_threshold = thresholds[best_threshold_idx]

    # tpr = tf.reduce_mean(tprs)
    # fpr = tf.reduce_mean(fprs)
    # accuracy = tf.reduce_mean(accuracy)
    # tpr = np.mean(tprs)
    # fpr = np.mean(fprs)
    # accuracy = np.mean(accuracy)
    return tprs, fprs, accuracy, best_threshold


def evaluate(embeddings1, embeddings2, actual_is_same):
    thresholds = np.arange(0, 4, 0.01)
    tprs, fprs, accuracy, best_threshold = calculate_roc(thresholds, embeddings1, embeddings2, actual_is_same)
    tpr = np.mean(tprs)
    fpr = np.mean(fprs)
    accuracy = np.mean(accuracy)
    return tpr, fpr, accuracy, best_threshold


if __name__ == '__main__':
    arcface = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    ckpt_path = tf.train.latest_checkpoint('./weights/arc_res50_ccrop')
    arcface.load_weights(ckpt_path)

    ref = np.load("data/kface_val_npy/references.npy").astype(np.float32) / 255.
    queries = np.load("data/kface_val_npy/queries.npy").astype(np.float32) / 255.
    is_same = np.load("data/kface_val_npy/is_same.npy")

    print(ref.shape, queries.shape)

    embeds1 = get_embedding(arcface, ref)
    embeds2 = get_embedding(arcface, queries)
    print(embeds1.shape, embeds2.shape)

    # tpr, fpr, accuracy, best_threshold = evaluate(embeds1, embeds2, is_same)
    thresholds = np.arange(1, 4, 0.01)
    tprs, fprs, _, _ = calculate_roc(thresholds, embeds1, embeds2, is_same)

    plt.plot(fprs, tprs)
