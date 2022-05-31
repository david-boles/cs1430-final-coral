from math import ceil
from random import randint, randrange
from subprocess import call
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import img_as_float32, io
from keras.callbacks import CSVLogger

# Number of distinct regions we have data for.
DATA_SECTIONS = 4
# Input shape will be (batch_sz, INPUT_DIM, INPUT_DIM, 3)
INPUT_DIM = 64
# Shift in pixels (1 dimension) from one input region to the next
TRAIN_EXAMPLE_STRIDE = ceil(INPUT_DIM / 1) # changed to 1
# Reduce the complexity of the model by increasing this number.
# Power of 2 between 1 and 16 inclusive.
COMPLEXITY_FACTOR = 16

MODEL_NAME = 'comp16_sz64_coralonly'

def load_dataset():
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    for sidx in range(DATA_SECTIONS):
        print (sidx)
        # Load
        section_input = img_as_float32(
            io.imread(f"./data/segmented-elisa-section_{sidx+1}-input.png"))
        section_output = img_as_float32(
            io.imread(f"./data/segmented-elisa-section_{sidx+1}-output.png")
        )

        # Shape
        section_shape = section_input.shape[:2]
        assert section_output.shape[:2] == section_shape

        # Input should be opaque, output should be opaque iff colored as coral
        assert (section_input[:, :, 3] == 1).all()
        # assert (
        #     (section_output[:, :, 3] == 1) == (section_output[:, :, 2] == 0.8352942)
        # ).all()
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


# Model block for the encoder.
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


# Model block for the decoder.
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


# Construct and compile the model.
def Segmenter():
    inputs = tf.keras.layers.Input(shape=[INPUT_DIM, INPUT_DIM, 3])

    down_stack = [
        downsample(64 / COMPLEXITY_FACTOR, 4, apply_batchnorm=False),
        downsample(128 / COMPLEXITY_FACTOR, 4),
        downsample(256 / COMPLEXITY_FACTOR, 4),
        downsample(512 / COMPLEXITY_FACTOR, 4),
        downsample(512 / COMPLEXITY_FACTOR, 4),
        downsample(512 / COMPLEXITY_FACTOR, 4),
        # downsample(512 / COMPLEXITY_FACTOR, 4),
        # (batch_size, 1, 1, 512 / COMPLEXITY_FACTOR)
    ]

    up_stack = [
        upsample(512 / COMPLEXITY_FACTOR, 4, apply_dropout=True),
        upsample(512 / COMPLEXITY_FACTOR, 4, apply_dropout=True),
        upsample(512 / COMPLEXITY_FACTOR, 4, apply_dropout=True),
        # upsample(512 / COMPLEXITY_FACTOR, 4),
        upsample(256 / COMPLEXITY_FACTOR, 4),
        upsample(128 / COMPLEXITY_FACTOR, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        1,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="sigmoid",
    )
    # (batch_size, 128, 128, 1)

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
        metrics=['accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )

    return model

"""
def calc_metrics(model, testX, testY):

    print(type(model))
    # predict crisp classes for test set
    yhat_classes = model.predict(testX, verbose=0)
    # reduce to 1d array
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testY, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(testY, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(testY, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testY, yhat_classes)
    print('F1 score: %f' % f1)

    metrics = (accuracy, precision, recall, f1)

    return metrics
"""

# Display the output for a random test image every batch
# and plot the accuracy every epoch.
class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, input, output, **args):
        super().__init__(**args)

        self.model = model
        self.input = input
        self.output = output

        self.train_accuracy = []
        self.val_accuracy = []

        self.img_figure = plt.figure()
        self.acc_figure = plt.figure()
        plt.show(block=False)

    def on_epoch_end(self, epoch, logs=None):
        self.train_accuracy.append(logs["accuracy"])
        self.val_accuracy.append(logs["val_accuracy"])

        plt.figure(self.acc_figure.number)
        plt.clf()
        plt.plot(self.train_accuracy)
        plt.plot(self.val_accuracy)
        plt.title("Accuracies by Epoch")
        plt.legend(["Train", "Validation"])


    def on_batch_end(self, batch, logs=None):
        idx = randrange(len(self.input))
        pred_mask = model.predict(self.input[[idx]])[0]

        titles = ["Input Image", "True Mask", "Predicted Mask"]
        images = [self.input[idx], self.output[idx], pred_mask]

        plt.figure(self.img_figure.number)
        plt.clf()
        for i, title, image in zip(range(3), titles, images):
            plt.subplot(1, 3, i + 1)
            plt.title(title)
            plt.imshow(image)
            plt.axis("off")
            
        plt.savefig('plots/train_img_batch'+str(batch)+'.png')
        plt.draw()
        plt.pause(0.01)
            

# Load data
train_input, train_output, test_input, test_output = load_dataset()
print(train_input.shape)

# Create file to save metrics
csv_logger = CSVLogger(MODEL_NAME+'.csv', append=True, separator=';')

# Define the model
model = Segmenter()
num_epochs = 15

# Train the model
history = model.fit(
    train_input,
    train_output,
    batch_size = 64,
    epochs=15,
    validation_data=(test_input, test_output),
    callbacks=[DisplayCallback(model, test_input, test_output), csv_logger]
)

# Plot the model's structure
# tf.keras.utils.plot_model(model, show_shapes=True)

# plot loss during training
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # plot accuracy during training
# plt.subplot(212)
# plt.title('Accuracy')
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show()

# serialize model to JSON
model_json = model.to_json()
with open(MODEL_NAME+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(MODEL_NAME + ".h5")
print("Saved model to disk")