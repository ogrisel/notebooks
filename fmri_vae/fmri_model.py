import numpy as np
from nilearn import datasets, image
from keras.layers import Conv3D, BatchNormalization, Flatten, Dense
from keras.layers import Dropout, Reshape, Conv3DTranspose, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def crop_5_8_8(data):
    return data[:, :5, :8, :8]


def make_models(input_shape=(40, 64, 64, 1), latent_dim=256,
                low_res_shape=(2, 2, 2, 128), dropout=0.2):
    encoder = Sequential([
        Conv3D(16, kernel_size=3, activation='relu',
               padding="same", input_shape=input_shape),
        BatchNormalization(),
        Conv3D(32, kernel_size=3, activation='relu',
               padding="same", strides=2),
        BatchNormalization(),
        Conv3D(32, kernel_size=3, activation='relu',
               padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=3, activation='relu',
               padding="same", strides=2),
        BatchNormalization(),
        Conv3D(64, kernel_size=3, activation='relu',
               padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=3, activation='relu',
               padding="same", strides=2),
        BatchNormalization(),
        Conv3D(128, kernel_size=3, activation='relu',
               padding="same", strides=2),
        BatchNormalization(),
        Conv3D(latent_dim, kernel_size=3, padding="same",
               strides=2, activation='relu'),
        Flatten(),
        Dropout(dropout),
        Dense(latent_dim),
    ], name="encoder")

    decoder = Sequential([
        Dense(np.prod(low_res_shape), input_shape=(latent_dim,)),
        Dropout(dropout),
        Reshape(low_res_shape),
        Conv3DTranspose(128, kernel_size=3, strides=2, activation='relu',
                        padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=3, activation='relu', padding="same"),
        BatchNormalization(),
        Conv3DTranspose(128, kernel_size=3, strides=2, activation='relu',
                        padding="same"),
        Lambda(function=crop_5_8_8),
        BatchNormalization(),
        Conv3D(64, kernel_size=3, activation='relu', padding="same"),
        BatchNormalization(),
        Conv3DTranspose(64, kernel_size=3, strides=2, activation='relu',
                        padding="same"),
        BatchNormalization(),
        Conv3D(32, kernel_size=3, activation='relu', padding="same"),
        BatchNormalization(),
        Conv3DTranspose(32, kernel_size=3, strides=2, activation='relu',
                        padding="same"),
        BatchNormalization(),
        Conv3D(16, kernel_size=3, activation='relu', padding="same"),
        BatchNormalization(),
        Conv3DTranspose(16, kernel_size=3, strides=2, activation='relu',
                        padding="same"),
        BatchNormalization(),
        Conv3D(1, kernel_size=3, activation=None, padding="same"),
    ], name="decoder")
    autoencoder = Sequential([encoder, decoder], name="autoencoder")
    return encoder, decoder, autoencoder



if __name__ == "__main__":
    data = datasets.fetch_haxby(subjects=(2,))
    fmri_filename = data.func[0]
    smoothed_img = image.smooth_img(fmri_filename, 2)
    
    smoothed_data = smoothed_img.get_data().transpose(3, 0, 1, 2)
    #mean = smoothed_data.mean(axis=0)
    #smoothed_data -= mean
    #scale = smoothed_data.std(axis=0) + 1e-6
    scale = smoothed_data.std()  # global scale
    smoothed_data /= scale
    smoothed_data = smoothed_data[:, :, :, :, None]
    input_shape = smoothed_data.shape[1:]
    smoothed_data_train = smoothed_data[:1200]
    smoothed_data_test = smoothed_data[1200:]
    
    encoder, decoder, autoencoder = make_models(input_shape=input_shape)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss="mse")

    
    filename = "haxby_autoencoder.{epoch:02d}-{val_loss:.4f}.hdf5"
    ckpt_cb = ModelCheckpoint(filename, monitor='val_loss',
                              verbose=1, save_best_only=False)
    filename = "haxby_autoencoder_best.hdf5"
    ckpt_best_cb = ModelCheckpoint(filename, monitor='val_loss',
                                   verbose=1, save_best_only=True)
    es_cb = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001,
                          verbose=1)
    lr_schedule_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                       cooldown=5, epsilon=0.0001, verbose=1)
    autoencoder.fit(smoothed_data_train, smoothed_data_train, 
                    validation_data=(smoothed_data_test, smoothed_data_test),
                    epochs=500, batch_size=32,
                    callbacks=[ckpt_cb, ckpt_best_cb, lr_schedule_cb, es_cb])