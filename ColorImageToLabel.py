import os
from pathlib import Path
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Reshape,
    UpSampling2D,
)
from keras.layers import LeakyReLU
from keras.callbacks import TensorBoard, ModelCheckpoint

from PythonUtils.file import unique_name
from PythonUtils.folder import get_abspath, create

from model.Kaggle_DeepLabV3Plus.deeplabv3plus import deeplabv3_plus
from model.abstract import CNN_model
from model.path import get_paths
from model.stage import stage

from generator.RGBInputGrayGroundTruthLabelSequences import DataSequence, ProcessingMode

# This is the generic class wrapper around the DeepLabV3 core class by incorporate Generator for image class data loading.
class DeepLabV3PlusCNN_I2D_O2D(CNN_model):
    """
    This model is used to provide semantic segmentation of images.
    Input: 2D images
    Output: 2D masks with LABEL.
    Output Type: Multiclass Label.
    """

    def __init__(self,
                 input_shape, output_classes, train_data_path: Path):

        self.input_shape = input_shape
        self.output_classes = output_classes

        self.train_data = None
        self.train_data_path: Path = train_data_path
        self.test_data = None # fixme: independent data used for testing (optional?)
        self.callbacks_list = None

        self.model = None

        self.path_prediction = None

        self.size_step = 256
        self.size_epoch = 500
        self.loss = "mae"  # fixme: loss parameters need to be redefined
        self.optimizer = "adam"
        self.metrics = ["mae", "mse", "mape", "cosine"]
        self.checkpoint_metric = "val_mean_absolute_error"
        self.checkpoint_metric_mode = "min"


        # Dynamically generate model input_path.
        this_file = os.path.realpath(__file__)
        project_root = get_abspath(this_file, 2)  # todo: this folder path is hard coded. Needs to be generalized.

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

            self.model = deeplabv3_plus(input_shape=self.input_shape, num_classes=self.output_classes)
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
        dummy_input = np.ones((100, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
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
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
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

        self.model.fit_generator(
            self.train_data,
            steps_per_epoch=size_step,
            epochs=size_epoch,
            validation_data=self.test_data,
            validation_steps=size_step,
            callbacks=self.callbacks_list,
        )
        self.path_prediction = os.path.join(self.path_model, unique_name() + ".h5")
        self.model.save(self.path_prediction)
        self.stage = stage.Ran
        return self.path_prediction

    def set_callbacks(self):
        """
        Two important callbacks: 1) model check points 2) tensorboard update.
        :return:
        """
        # Model name.
        name_model_checkpoint = os.path.join(self.path_model, f"{unique_name()}_{__name__}")

        # Checkpoint 1
        callback_save_model = ModelCheckpoint(
            name_model_checkpoint,
            monitor=self.checkpoint_metric,
            verbose=1,
            save_best_only=True,
            mode=self.checkpoint_metric_mode,
        )

        # Checkpoint 2
        # Generate the tensorboard
        callback_tensorboard = TensorBoard(
            log_dir=self.path_log_run,
            histogram_freq=0,
            write_images=True
        )

        self.callbacks_list = [callback_tensorboard, callback_save_model]