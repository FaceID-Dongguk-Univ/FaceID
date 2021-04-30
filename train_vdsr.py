import tensorflow as tf
import datetime
import cv2
from glob import glob

from networks.resolution.vdsr import vdsr
from metrics import PSNR, SSIM

train_list = glob("drive/MyDrive/faceID/data/kface/*/*.jpg")
val_list = glob("drive/MyDrive/faceID/data/kface_val/*/*.jpg")

PIXEL = 112
INPUT_SIZE = (PIXEL // 2, PIXEL // 2, 3)
TARGET_SIZE = (PIXEL, PIXEL, 3)


def get_data():
    for img_path in train_list:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            resized = cv2.resize(img, INPUT_SIZE[:2])
            bicubic = cv2.resize(resized, TARGET_SIZE[:2])
            yield bicubic, cv2.resize(img, TARGET_SIZE[:2])
        except GeneratorExit:
            return


def get_val_data():
    for img_path in val_list:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            resized = cv2.resize(img, INPUT_SIZE[:2])
            bicubic = cv2.resize(resized, TARGET_SIZE[:2])
            yield bicubic, cv2.resize(img, TARGET_SIZE[:2])
        except GeneratorExit:
            return


def main():

    train_ds = tf.data.Dataset.from_generator(
        get_data,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([PIXEL, PIXEL, 3]),
                       tf.TensorShape([PIXEL, PIXEL, 3]))
    )

    val_ds = tf.data.Dataset.from_generator(
        get_val_data,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([PIXEL, PIXEL, 3]),
                       tf.TensorShape([PIXEL, PIXEL, 3]))
    )

    train_ds = train_ds.shuffle(buffer_size=100)

    # model define
    model = vdsr(TARGET_SIZE, 64)

    def scheduler(epoch):
        if epoch < 20:
            return base_lr
        elif epoch < 40:
            return base_lr / 10
        elif epoch < 60:
            return base_lr / 100
        else:
            return base_lr / 1000

    base_lr = 0.01
    optimizer = tf.keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=0.0001)
    epochs = 80
    batch_size = 64
    model_name = f"vdsr-bs{batch_size}-ps{PIXEL}"
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + model_name + '/fit'

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('weights/' + model_name + '.h5',
                                                          save_best_only=True, verbose=1)
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

    callbacks = [lr_scheduler, model_checkpoint, tensor_board]

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=[PSNR, SSIM])

    model.fit(
        train_ds.batch(batch_size),
        validation_data=val_ds.batch(batch_size),
        epochs=epochs,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
