from model.Kaggle_DeepLabV3Plus.ModelClassSpec import DeepLabV3PlusCNN_I2D_O2D
from pathlib import Path
from lossfn.ssim import loss_SSIM, loss_mae_diff_SSIM_composite, loss_mse_diff_SSIM_composite
from lossfn.f1 import f1_metric

from model.cleanup import cleanLog

# This is the main script entry point to invoke the CNN.

# By importing from different class as model, they can be invoked individually here.
# Setup:
batch_size = 8  # images PER epoch

input_shape = (
    240,  # height first
    320,  # width later.
    3,  # color channel
)  # Train images are RGB. first number is the number of rows (y positions), second number is the number of columns (x)

output_channel = (
    240,  # height
    320,  # width
    1,  # color channel
)  # Label images are Grayscale

num_classes = 1

size_step = 8
size_epoch = 2000
train_data_path = Path(
    r"C:\Git\MarkerTrainer\data_trailmix\augmentation_2019-09-21T22_07_00.474870"
)  # this folder MUST contain a LABEL folder and a TRAIN folder of flat images WITH IDENTICAL NAME-label pair.

# Model creation:
model_single_class = DeepLabV3PlusCNN_I2D_O2D(
    input_shape=input_shape,
    output_classes=num_classes,
    train_data_path=train_data_path,
    loss="mse",
    metrics=["mae",
             "mse",
             "mape",
             "cosine",
             loss_SSIM,
             loss_mae_diff_SSIM_composite,
             loss_mse_diff_SSIM_composite,
             f1_metric],
    checkpoint_metric="val_loss",
)
model_single_class.create()
model_single_class.compile()
model_single_class.IOCheck()
model_single_class.load_data(batch_size=batch_size)
final_model1, final_model1_weights = model_single_class.run(size_step, size_epoch)
model_single_class.predict(final_model1_weights, r"C:\Git\MarkerTrainer\data_test_results_2019-09-22T013003EST")
