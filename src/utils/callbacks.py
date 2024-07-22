import tensorflow as tf
import numpy as np
import time
import os
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau,ModelCheckpoint

""" here in this template we are """

def get_timestamp(name):
    timestamp = time.asctime().replace(" ","_").replace(":","_")
    unique_name = f"{name}_at_{timestamp}"



    return unique_name


""" Here we have defined all the callbacks and tensorboard """
def get_callbacks(config):
    logs = config["log"]
    unique_dir_name = get_timestamp("tb_log")
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs["log_dir"],logs["TENSORBOARD_ROOT_LOG_DIR"],unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR,exist_ok=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)
   # file_writer = tf.summary.create_file_writer(logdir = TENSORBOARD_ROOT_LOG_DIR)
    LR_Reduce_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')
    artifacts = config["artifacts"]
    CKPT_dir = os.path.join(artifacts["artifacts_dir"],artifacts["CHECKPOINT_DIR"])
    os.makedirs(CKPT_dir,exist_ok=True)
    

    CKPT_path1 = os.path.join(CKPT_dir, "model_ckpt1.h5")
    CKPT_path2 = os.path.join(CKPT_dir, "model_ckpt2.h5")
    model_ckpt1 = ModelCheckpoint(CKPT_path1, monitor='val_loss', save_weights_only=False,save_best_only=True, period=1)
    model_ckpt2 = ModelCheckpoint(CKPT_path2, monitor='val_loss', save_weights_only=True,save_best_only=True, period=1)

    return [tensorboard_cb,LR_Reduce_callback,model_ckpt1,model_ckpt2]