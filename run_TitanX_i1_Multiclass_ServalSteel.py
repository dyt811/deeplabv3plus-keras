from model.Kaggle_DeepLabV3Plus.ModelClassSpec import DeepLabV3PlusCNN_I2D_O2D
from pathlib import Path
from lossfn.ssim import loss_SSIM, loss_mae_diff_SSIM_composite, loss_mse_diff_SSIM_composite
from lossfn.f1 import f1_metric

"""
# This is the main class to run MULTI class training for ServeStall. Labelled data ideally should be in grayscale label 1200 x 255 x 3  
"""
# By importing from different class as model, they can be invoked individually here.
# Setup:
batch_size = 2  # images PER epoch

input_shape = (
    256,  # height first
    1600,  # width later.
    3,  # color channel
)  # Train images are RGB. first number is the number of rows (y positions), second number is the number of columns (x)

output_channel = (
    256,  # height
    1600,  # width
    4,  # color channel
)  # Label images are Grayscale

num_classes = 4
size_step = 4
size_epoch = 7500
size_epoch = 7500

train_data_path = Path(r"C:\Git\MarkerTrainer\data_servestal\labelled_images")  # this folder MUST contain a LABEL folder and a TRAIN folder of flat images WITH IDENTICAL NAME-label pair.

# Model creation:
model_multi_class = DeepLabV3PlusCNN_I2D_O2D(
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

model_multi_class.create()
model_multi_class.compile()
model_multi_class.IOCheck()
model_multi_class.load_data(batch_size=batch_size)
final_model1, final_model1_weights = model_multi_class.run(size_step, size_epoch)

# Update this folder path to the folder contain HOLDOUT images.
#model_multi_class.predict(final_model1_weights, r"C:\Git\MarkerTrainer\data_test_results_2019-09-22T013003EST")

