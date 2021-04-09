import tensorflow as tf
import datetime
import cv2
from glob import glob

from networks.resolution.vdsr import vdsr
from metrics import PSNR, SSIM


if __name__ == "__main__":

    # prepare dataset
    train_list = glob("./lfw-deepfunneled/*/*.jpg")
    INPUT_SIZE = (100, 100, 3)
    TARGET_SIZE = (250, 250, 3)


    def get_data():
        for img_path in train_list:
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                resized = cv2.resize(img, INPUT_SIZE[:2])
                bicubic = cv2.resize(resized, TARGET_SIZE[:2])

                yield bicubic, img

            except GeneratorExit:
                return


    ds = tf.data.Dataset.from_generator(
        get_data,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([TARGET_SIZE[0], TARGET_SIZE[1], 3]),
                       tf.TensorShape([TARGET_SIZE[0], TARGET_SIZE[1], 3]))
    )

    # train-validation split
    DATA_SIZE = len(train_list)
    TRAIN_SIZE = int(0.7 * DATA_SIZE)
    ds = ds.shuffle(buffer_size=1024)
    train_ds = ds.take(TRAIN_SIZE)
    val_ds = ds.skip(TRAIN_SIZE)

    # model define
    model = vdsr(TARGET_SIZE, 64)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=[PSNR, SSIM])

    # model fit
    weight_path = "./weights/vdsr-test_epochs50.hdf5"
    log_path = "log\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model.fit(
        ds.batch(32), epochs=50,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(weight_path, verbose=1, save_best_only=True),
                   tf.keras.callbacks.TensorBoard(log_path, historam_freq=1)]
    )
