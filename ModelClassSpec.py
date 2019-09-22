import os
from pathlib import Path
from keras.callbacks import TensorBoard, ModelCheckpoint

from PythonUtils.PUFile import unique_name
from PythonUtils.PUFolder import get_abspath, create

from model.Kaggle_DeepLabV3Plus.ModelLayersSpec import deeplabv3_plus
from model.abstract import CNN_model
from model.path import get_paths
from model.stage import stage

from generator.RGBInputGrayGroundTruthLabelSequences import DataSequence, ProcessingMode
from lossfn.ssim import loss_SSIM, loss_mae_diff_SSIM_composite, loss_mse_diff_SSIM_composite
from lossfn.f1 import f1_metric
from model.Kaggle_DeepLabV3Plus.predict_mask import predict_folder

import keras.backend as K
# Force Keras to use 16 bits to free up more memory at the expense of training time.
dtype = "float16"
#K.set_floatx(dtype)

# This is the generic class wrapper around the DeepLabV3 core class by incorporate Generator for image class data loading.
class DeepLabV3PlusCNN_I2D_O2D(CNN_model):
    """
    This model is used to provide semantic segmentation of images.
    Input: 2D images
    Output: 2D masks with LABEL.
    Output Type: Multiclass Label.
    """

    def __init__(self,
                 input_shape,
                 output_classes,
                 train_data_path: Path,
                 optimizer="adam",
                 loss=loss_mae_diff_SSIM_composite,
                 metrics=["mae", "mse", "mape", "cosine", loss_SSIM, loss_mae_diff_SSIM_composite, loss_mse_diff_SSIM_composite],
                 checkpoint_metric="val_loss_mse_diff_SSIM_composite",
                 checkpoint_metric_mode="min",
                 ):
        # Use these settings per constructor input.
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_metric_mode = checkpoint_metric_mode

        self.train_data = None
        self.train_data_path: Path = train_data_path
        self.test_data = None  # fixme: independent data used for testing (optional?)
        self.callbacks_list = None

        self.model = None
        self.path_prediction = None

        # Default step and epoch size.   Easily overwritten during the run stage.
        self.size_step = 256
        self.size_epoch = 500

        # Dynamically generate model input_path.
        this_file = os.path.realpath(__file__)
        project_root = get_abspath(
            this_file, 2
        )  # todo: this folder path is hard coded. Needs to be generalized.

        # Log path.
        self.path_log, self.path_model = get_paths(project_root)

        # Log run path.
        self.path_log_run = os.path.join(self.path_log, unique_name() + __name__)

        # Create the Log run path.
        create(self.path_log_run)

        self.model_stage = stage.Initialized

    def create(self):
        """
        This model is optimizd for REGRESSION type of network no a
        :param input_shape:
        :param output_classes:
        :return:
        """
        if self.model_stage.value < stage.Created.value:

            self.model = deeplabv3_plus(
                input_shape=self.input_shape, num_classes=self.output_classes
            )
            self.model_stage = stage.Created
        else:
            print("Stage: Model already created.")
        return self.model

    def IOCheck(self):
        """
        A quick but important sanity check step to ensure the input and output shapes conforms to the public's expectations.
        :return:
        """
        import numpy as np

        dummy_input = np.ones(
            (100, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        )
        preds = self.model.predict(dummy_input)

        print(preds.shape)

    def compile(self):
        """
        This is a model specific compilation process, must choosen/update based on the purpose of the network.
        :param model:
        :return:
        """
        if self.model_stage != stage.Created:
            print("Stage: Model not already created. Check where you are at. ")
            return self.model

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics
        )
        self.model.summary()
        self.model_stage = stage.Compiled

        return self.model

    def load_data(self, data_path="", batch_size=5):
        """
        Load the data specification from the path in the .env.
        """
        if data_path == "":
            data_path = self.train_data_path

        if self.model_stage != stage.Compiled:
            print("Stage: model has not been compiled yet.")
            return

        # Path to the train.
        path_train_spec = data_path

        # Generate data sequence.
        train_data = DataSequence(
            path_train_spec, batch_size, mode=ProcessingMode.Train
        )

        # Path to the validation.
        path_validate_spec = data_path

        # Generate data sequence.
        validation_data = DataSequence(
            path_validate_spec, batch_size, mode=ProcessingMode.Validation
        )

        self.train_data = train_data
        self.test_data = validation_data
        self.model_stage = stage.DataLoaded

    def run(self, size_step=None, size_epoch=None):
        """
        Actually execute the pipeline if all stages are good.
        :param size_step:
        :param size_epoch:
        :return:
        """
        if self.model_stage != stage.DataLoaded:
            print("Stage: Data model has not been loaded yet")
            return None

        # Update parameter if never received them in the first place.
        if size_step is None:
            size_step = self.size_step
        if size_epoch is None:
            size_epoch = self.size_epoch

        # Write out the import paramemters used for the model BEFORE actual doing the training
        self.record_settings()

        # Set the proper call back to write functions and record results for tensorboard.
        self.set_callbacks()

        self.model.fit_generator(
            self.train_data,
            steps_per_epoch=size_step,
            epochs=size_epoch,
            validation_data=self.test_data,
            validation_steps=size_step,
            callbacks=self.callbacks_list,
        )

        # Timestamp and save the final model as well as its weights
        timestamp = unique_name()

        name_final_model = os.path.join(self.path_model, f"{timestamp}_FinalModel_{__name__}.h5")
        self.model.save(name_final_model)

        name_final_model_weights = os.path.join(self.path_model, f"{timestamp}_FinalModelWeights_{__name__}.h5")
        self.model.save_weights(name_final_model_weights)

        self.stage = stage.Ran
        return name_final_model, name_final_model_weights

    def record_settings(self):
        """
        An important documentation functions that keeps track of the key parameters in a json files for later review.
        :return:
        """
        import json
        dict_setting = {}
        dict_setting["input_shape"] = self.input_shape
        dict_setting["output_classes"] = self.output_classes
        dict_setting["loss"] = self.loss
        dict_setting["optimizer"] = self.optimizer
        dict_setting["metrics"] = self.metrics
        dict_setting["checkpoint_metric"] = self.checkpoint_metric
        dict_setting["checkpoint_metric_mode"] = self.checkpoint_metric_mode
        dict_setting["train_data_path"] = self.train_data_path
        dict_setting["path_prediction"] = self.path_prediction

        # Default step and epoch size.   Easily overwritten during the run stage.
        dict_setting["size_step"] = self.size_step
        dict_setting["size_epoch"] = self.size_epoch

        # The path where json is saved is in the model folder
        path_json = Path(self.path_model) / f"{unique_name()}_ModelParametersSpecification_{__name__}.json"
        with open(path_json, 'w') as outfile:
            json.dump(dict_setting, outfile)

    def predict(self, path_model_weights, path_test):
        """
        Call the prediction function to predict the final model using weights. 
        :param path_test:
        :return:
        """
        predict_folder(path_model_weights, path_test)


    def set_callbacks(self):
        """
        Two important callbacks: 1) model check points 2) tensorboard update.
        :return:
        """
        # Model name.
        model_name = unique_name()

        checkpoint_last_best_model = os.path.join(self.path_model, f"{model_name}_LastBest_{__name__}.h5")
        checkpoint_last_model_weight = os.path.join(self.path_model, f"{model_name}_Weights_{__name__}.h5")
        checkpoint_last_best_model_weight = os.path.join(self.path_model, f"{model_name}_LastBestWeights_{__name__}.h5")

        # Checkpoint for saving the LAST BEST MODEL.
        callback_save_best_model = ModelCheckpoint(
            checkpoint_last_best_model,
            monitor=self.checkpoint_metric,
            verbose=1,
            save_best_only=True,
            mode=self.checkpoint_metric_mode,
        )

        # Checkpoint for saving the LAST BEST MODEL WEIGHTS only without saving the full model.
        callback_save_best_model_weights = ModelCheckpoint(
            checkpoint_last_best_model_weight,
            monitor=self.checkpoint_metric,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode=self.checkpoint_metric_mode,
        )

        # Checkpoint for saving the LATEST MODEL WEIGHT.
        callback_save_model_weights = ModelCheckpoint(
            checkpoint_last_model_weight,
            verbose=1,
            save_weights_only=True,
        )

        # Checkpoint for updating the tensorboard
        callback_tensorboard = TensorBoard(
            log_dir=self.path_log_run,
            histogram_freq=0,
            write_images=True
        )

        self.callbacks_list = [
            callback_tensorboard,  # always update the tensorboard
            callback_save_model_weights,  # always save model weights.
            callback_save_best_model,
            callback_save_best_model_weights,
        ]