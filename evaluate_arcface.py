"""
An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition
https://arxiv.org/abs/1801.07698

This ArcFace code is based on peteryuX's arcface-tf2.
https://github.com/peteryuX/arcface-tf2
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import l2_norm, load_yaml
from networks.recognition.models import ArcFaceModel
from networks.resolution import srgan, vdsr


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
    # precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_is_same):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    tprs = np.zeros((len(thresholds)))
    fprs = np.zeros((len(thresholds)))
    accuracy = np.zeros((len(thresholds)))
    # precision = np.zeros((len(thresholds)))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), axis=1)

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], accuracy[threshold_idx] = calculate_accuracy(threshold, dist, actual_is_same)
    best_threshold_idx = np.argmax(accuracy)
    best_threshold = thresholds[best_threshold_idx]

    return tprs, fprs, accuracy, best_threshold


def evaluate(embeddings1, embeddings2, actual_is_same):
    thresholds = np.arange(0, 4, 0.01)
    tprs, fprs, accuracy, best_threshold = calculate_roc(thresholds, embeddings1, embeddings2, actual_is_same)
    tpr = np.mean(tprs)
    fpr = np.mean(fprs)
    accuracy = np.max(accuracy)
    return tpr, fpr, accuracy, best_threshold


if __name__ == '__main__':
    import os
    import cv2
    from sklearn.metrics import auc

    arcface = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    ckpt_path = tf.train.latest_checkpoint('./weights/arc_res50_kface_finetune_9K-lr0.001-bs128-epochs50')
    arcface.load_weights(ckpt_path)

    data_path = "data/kface_val_npy"

    ref = np.load(os.path.join(data_path, "references.npy")).astype(np.float32) / 255.
    queries = np.load(os.path.join(data_path, "queries.npy")).astype(np.float32) / 255.
    is_same = np.load(os.path.join(data_path, "is_same.npy"))

    print(ref.shape, queries.shape)

    ### srgan
    # cfg = load_yaml('configs/srgan.yaml')
    # srgan = srgan.FastSRGAN(cfg).build_generator()
    # ckpt_path = tf.train.latest_checkpoint('weights/srgan/generator')
    # srgan.load_weights(ckpt_path)
    # ref = cv2.resize(ref, (56, 56))
    # ref = srgan.predict(ref)
    ### srgan

    ### vdsr
    vdsr = vdsr.vdsr((112, 112, 3), 64)
    vdsr.load_weights("weights/vdsr-bs64-ps112.hdf5")
    ref = vdsr.predict(ref)
    ### vdsr
    print(ref.shape, queries.shape)

    embeds1 = get_embedding(arcface, ref)
    embeds2 = get_embedding(arcface, queries)
    print(embeds1.shape, embeds2.shape)

    # tpr, fpr, accuracy, best_threshold = evaluate(embeds1, embeds2, is_same)
    thresholds = np.arange(0, 3, 0.01)
    tprs, fprs, accuracy, best_threshold = calculate_roc(thresholds, embeds1, embeds2, is_same)
    accuracy = np.max(accuracy)
    print(f"Maximum Accuracy: {float(accuracy):.4f}, Best Threshold: {best_threshold}")

    plt.figure(figsize=(5, 5))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Fine tuned on [K-Face])")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(fprs, tprs, label=f"ROC Curve (AUC = {auc(fprs, tprs):.4f}")
    plt.legend()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.savefig('resource/epochs100.png')
