# -----
# An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition
# https://arxiv.org/abs/1801.07698
#
# This ArcFace code is based on 4uiiurz1's keras-arcface.
# https://github.com/4uiiurz1/keras-arcface
# -----

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime

from networks.recognition.resnet50_arcface import resnet50_arcface


if __name__ == "__main__":

    # prepare lfw dataset
    n_classes = 5749
    IMG_SIZE = (250, 250, 3)
    BATCH_SIZE = 32

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.3
    )

    train_ds = generator.flow_from_directory(
        directory='./lfw-deepfunneled',
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(250, 250),
        class_mode='categorical',
        subset='training'
    )

    validation_ds = generator.flow_from_directory(
        directory='./lfw-deepfunneled',
        batch_size=BATCH_SIZE,
        target_size=(250, 250),
        class_mode='categorical',
        subset='validation'
    )


    def data_generator(dataset):
        while True:
            (img, label) = dataset.next()
            yield (img, label), label


    # model define
    model = resnet50_arcface(IMG_SIZE, n_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # model fit
    log_path = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-resnet50-arcface-epochs20"
    weight_path = f"/weights/{datetime.datetime.now().strftime('%m%d-%H')}-resnet50-arcface-epochs20.hdf5"

    model.fit(
        data_generator(train_ds),
        steps_per_epoch=train_ds.samples // BATCH_SIZE,
        validation_data=data_generator(validation_ds),
        validation_steps=validation_ds.samples // BATCH_SIZE,
        epochs=20,
        callbacks=[ModelCheckpoint(weight_path, verbose=1, save_best_only=True),
                   TensorBoard(log_path, histogram_freq=1)]
    )
