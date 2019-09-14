from model.Kaggle_DeepLabV3Plus.ColorImageToLabel import (
    DeepLabV3PlusCNN_I2D_O2D
)
from pathlib import Path

from model.cleanup import cleanLog

# This is the main script entry point to invoke the CNN.

# By importing from different class as model, they can be invoked individually here.
# Setup:
batch_size = 32 # 32 images PER epoch
input_shape = (1600, 256, 3)  # this is the input shape of the FINAL TRIMMed model.
output_channel = (1600, 256, 1)
num_classes = 4
size_step = 128
size_epoch = 500
train_data_path = Path(r"C:\Git\MarkerTrainer\data_1\labelled_images") # this folder MUST contain a LABEL folder and a TRAIN folder of flat images WITH IDENTICAL NAME-label pair.
# Train images are RGB
# Label images are Grayscale

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
