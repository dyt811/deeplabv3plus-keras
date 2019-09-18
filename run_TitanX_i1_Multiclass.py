from model.Kaggle_DeepLabV3Plus.ModelClassSpec import (
    DeepLabV3PlusCNN_I2D_O2D
)
from pathlib import Path

from model.cleanup import cleanLog

# This is the main script entry point to invoke the CNN.

# By importing from different class as model, they can be invoked individually here.
# Setup:
batch_size = 8 # images PER epoch
input_shape = (240, 320, 3)  # Train images are RGB. first number is the number of rows (y positions), second number is the number of columns (x)
output_channel = (240, 320, 1)  # Label images are Grayscale
num_classes = 1 # output one class PER image
size_step = 8
size_epoch = 50000
train_data_path = Path(r"C:\Git\MarkerTrainer\data_validation1\augmentation_2019-09-17T01_21_43.765359")  # this folder MUST contain a LABEL folder and a TRAIN folder of flat images WITH IDENTICAL NAME-label pair.

# Model creation:
model1 = DeepLabV3PlusCNN_I2D_O2D(
    input_shape=input_shape,
    output_classes=num_classes,
    train_data_path=train_data_path)
model1.create()
model1.compile()
model1.IOCheck()
model1.load_data(batch_size=batch_size)
model1.run(size_step, size_epoch)
