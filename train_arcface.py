"""
An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition
https://arxiv.org/abs/1801.07698

This ArcFace code is based on 4uiiurz1's keras-arcface.
https://github.com/4uiiurz1/keras-arcface
"""
import tensorflow as tf

from networks.recognition.models import ArcFaceModel, ArcHead
from networks.recognition.losses import SoftmaxLoss
import networks.recognition.dataset as dataset
from utils import load_yaml


def main(cfg_path):
    """
    train model
    fine tune from MS-Celeb to k-face
    """

    """
    1. load pretrained model
    """
    cfg = load_yaml(cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True)
    model.summary(line_length=80)

    ckpt_path = tf.train.latest_checkpoint('./weights/arc_res50_ccrop')
    print("[*] load ckpt from {}".format(ckpt_path))
    model.load_weights(ckpt_path)

    """ 
    2. fine tuning
    """
    dataset_len = 21600
    steps_per_epoch = dataset_len // cfg['batch_size']
    epochs, steps = 1, 1

    n_label = tf.keras.layers.Input([])

    n_model = model.layers[1](model.layers[0].output)
    n_model = model.layers[2](n_model)
    n_model = ArcHead(num_classes=400, margin=0.5, logist_scale=64.0)(n_model, n_label)
    n_model = tf.keras.Model(inputs=(model.input[0], n_label), outputs=n_model)

    model = n_model
    for layer in model.layers[:2]:
        layer.trainable = False

    model.summary()

    """ 
    3. train
    """
    # compile
    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()

    # load dataset
    train_dataset = dataset.load_tfrecord_dataset(cfg['train_dataset'],
                                                  cfg['batch_size'],
                                                  cfg['binary_img'],
                                                  is_ccrop=cfg['is_ccrop'])
    train_dataset = iter(train_dataset)

    # logging
    model_name = f"-lr{cfg['base_lr']}-e{cfg['epochs']}-trainable2"
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, 'weights/' + cfg['sub_name'] + model_name, max_to_keep=3)
    summary_writer = tf.summary.create_file_writer("logs/" + cfg['sub_name'] + model_name)

    # training loop
    while epochs <= cfg['epochs']:
        inputs, labels = next(train_dataset)

        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            reg_loss = tf.reduce_sum(model.losses)
            pred_loss = loss_fn(labels, logits)
            total_loss = pred_loss + reg_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if steps % 135 == 0:
            verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
            print(verb_str.format(epochs, cfg['epochs'],
                                  steps % steps_per_epoch,
                                  steps_per_epoch,
                                  total_loss.numpy(),
                                  learning_rate.numpy()))

            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss/total loss', total_loss, step=steps)
                tf.summary.scalar(
                    'loss/pred loss', pred_loss, step=steps)
                tf.summary.scalar(
                    'loss/reg loss', reg_loss, step=steps)
                tf.summary.scalar(
                    'learning rate', optimizer.lr, step=steps)

        ckpt.step.assign_add(1)
        if int(ckpt.step) % cfg['save_steps'] == 0:
            save_path = ckpt_manager.save()
            print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")

        steps += 1
        epochs = steps // steps_per_epoch + 1


if __name__ == "__main__":
    main("configs/arc_res50_kface_finetune.yaml")
