from math import ceil
from random import randint, randrange
from subprocess import call
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import img_as_float32, io

# Number of distinct regions we have data for.
DATA_SECTIONS = 3
# Input shape will be (batch_sz, INPUT_DIM, INPUT_DIM, 3)
INPUT_DIM = 128
# Shift in pixels (1 dimension) from one input region to the next
TRAIN_EXAMPLE_STRIDE = ceil(INPUT_DIM / 5)


def load_dataset():
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    for sidx in range(DATA_SECTIONS):
        # Load
        section_input = img_as_float32(io.imread(f"./data/section-{sidx+1}-input.png"))
        section_output = img_as_float32(
            io.imread(f"./data/section-{sidx+1}-output.png")
        )

        # Shape
        section_shape = section_input.shape[:2]
        assert section_output.shape[:2] == section_shape

        # Input should be opaque, output should be opaque iff colored as coral
        assert (section_input[:, :, 3] == 1).all()
        assert (
            (section_output[:, :, 3] == 1) == (section_output[:, :, 2] == 0.8352942)
        ).all()
        assert ((section_output[:, :, 3] == 0) == (section_output[:, :, 2] == 0)).all()

        # Drop/keep only alpha channel
        section_input = section_input[:, :, :3]
        section_output = section_output[:, :, 3]

        # Split into examples. Training is taken from the first row.
        # Test set has no overlap, training set does.
        for row in range(
            INPUT_DIM, section_shape[0] - INPUT_DIM + 1, TRAIN_EXAMPLE_STRIDE
        ):
            for col in range(0, section_shape[1] - INPUT_DIM + 1, TRAIN_EXAMPLE_STRIDE):
                train_input.append(
                    section_input[row : row + INPUT_DIM, col : col + INPUT_DIM, :]
                )
                train_output.append(
                    section_output[row : row + INPUT_DIM, col : col + INPUT_DIM]
                )
        for col in range(0, section_shape[1] - INPUT_DIM + 1, INPUT_DIM):
            test_input.append(section_input[0:INPUT_DIM, col : col + INPUT_DIM, :])
            test_output.append(section_output[0:INPUT_DIM, col : col + INPUT_DIM])

    return (np.stack(x) for x in (train_input, train_output, test_input, test_output))


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Segmenter():
    inputs = tf.keras.layers.Input(shape=[INPUT_DIM, INPUT_DIM, 3])

    down_stack = [
        downsample(
            64 / 16, 4, apply_batchnorm=False
        ),  # (batch_size, INPUT_DIM/2, INPUT_DIM/2, 64)
        downsample(128 / 16, 4),  # (batch_size, INPUT_DIM/4, INPUT_DIM/4, 128)
        downsample(256 / 16, 4),  # (batch_size, INPUT_DIM/8, INPUT_DIM/8, 256)
        downsample(512 / 16, 4),  # (batch_size, INPUT_DIM/16, INPUT_DIM/16, 512)
        downsample(512 / 16, 4),  # (batch_size, INPUT_DIM/32, INPUT_DIM/32, 512)
        downsample(512 / 16, 4),  # (batch_size, INPUT_DIM/64, INPUT_DIM/64, 512)
        downsample(512 / 16, 4),  # (batch_size, INPUT_DIM/128, INPUT_DIM/128, 512)
        # downsample(512, 4),  # (batch_size, INPUT_DIM/256, INPUT_DIM/256, 512)
    ]

    up_stack = [
        upsample(512 / 16, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512 / 16, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512 / 16, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512 / 16, 4),  # (batch_size, 16, 16, 1024)
        upsample(256 / 16, 4),  # (batch_size, 32, 32, 512)
        upsample(128 / 16, 4),  # (batch_size, 64, 64, 256)
        # upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        1,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, ?, ?, 1)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


# # TODO
# def display(display_list):
#     plt.close()
#     plt.figure(figsize=(15, 15))

#     title = ["Input Image", "True Mask", "Predicted Mask"]

#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i + 1)
#         plt.title(title[i])
#         plt.imshow(display_list[i])
#         plt.axis("off")
#     plt.show()
#     pass


# def show_predictions_builder(model, inputs, outputs):
#     def callback():
#         idx = randint(len(inputs))
#         pred_mask = model.predict(inputs[idx])
#         display([inputs[idx], outputs[idx], pred_mask])

#     return callback


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, input, output, **args):
        super().__init__(**args)

        self.figure = plt.figure(figsize=(15, 15))
        plt.show(block=False)
        self.model = model
        self.input = input
        self.output = output

    # def on_epoch_end(self, epoch, logs=None):
    #     # clear_output(wait=True)
    #     idx = randrange(len(test_input))
    #     pred_mask = model.predict(test_input[[idx]])
    #     display([test_input[idx], test_output[idx], pred_mask[0]])
    #     # print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

    def on_batch_end(self, batch, logs=None):
        # idx = randrange(len(test_input))
        # pred_mask = model.predict(test_input[[idx]])
        # display([test_input[idx], test_output[idx], pred_mask[0]])

        idx = randrange(len(self.input))
        pred_mask = model.predict(self.input[[idx]])[0]

        titles = ["Input Image", "True Mask", "Predicted Mask"]
        images = [self.input[idx], self.output[idx], pred_mask]

        plt.figure(self.figure.number)
        plt.clf()
        for i, title, image in zip(range(3), titles, images):
            plt.subplot(1, 3, i + 1)
            plt.title(title)
            plt.imshow(image)
            plt.axis("off")
        plt.draw()
        plt.pause(0.01)
        pass  # TODO


train_input, train_output, test_input, test_output = load_dataset()
model = Segmenter()

model.fit(
    train_input,
    train_output,
    # test_input[:10],  # TODO
    # test_output[:10],
    batch_size=100,  # TODO
    epochs=1000,
    validation_data=(test_input, test_output),
    callbacks=[DisplayCallback(model, test_input, test_output)],
)
