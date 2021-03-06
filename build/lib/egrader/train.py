import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from egrader.bert_util import create_learning_rate_scheduler

log_dir = 'log/bert' + datetime.now().strftime("%Y%m%d-%H%M%s")


def train_movie_review(model, data):
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    total_epoch_count = 40
    # model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,
    model.fit(x=data.train_x, y=data.train_y,
              validation_split=0.1,
              batch_size=48,
              shuffle=True,
              epochs=total_epoch_count,
              callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                        end_learn_rate=1e-7,
                                                        warmup_epoch_count=20,
                                                        total_epoch_count=total_epoch_count),
                         keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                         tensorboard_callback])

    model.save_weights('data/trained_model/movie_reviews.h5', overwrite=True)
    return model


def train_essay_grader(model, data, checkpoint_path, epoch=40, batch_size=16, validation_split=0.1):
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    filepath = checkpoint_path+"/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,
    model.fit(x=data.train_x, y=data.train_y,
              validation_split=validation_split,
              batch_size=batch_size,
              shuffle=True,
              epochs=epoch,
              callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                        end_learn_rate=1e-7,
                                                        warmup_epoch_count=20,
                                                        total_epoch_count=epoch),
                         keras.callbacks.EarlyStopping(patience=epoch/2, restore_best_weights=True),
                         tensorboard_callback,
                         checkpoint])
    return model
