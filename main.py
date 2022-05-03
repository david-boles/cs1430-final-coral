from math import ceil
import matplotlib.pyplot as plt
import numpy as np
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

    return (train_input, train_output, test_input, test_output)


load_dataset()
