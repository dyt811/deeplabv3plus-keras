from keras.preprocessing import image
import os
import numpy as np


from PythonUtils.PUFolder import recursive_list
from PythonUtils.PUFile import unique_name
from PythonUtils.PUJson import write_json
from model.Kaggle_DeepLabV3Plus.ModelLayersSpec import deeplabv3_plus
from PIL import Image

from tqdm import tqdm
from pathlib import Path


def predict_image(path_input_model_weight: str, input_image):
    """
    Use the provided model weight to predice the image segmentation output
    :param path_input_model_weight:
    :param input_image:
    :return:
    """

    # predicting multiple images at once
    img = image.load_img(
        input_image,
        # target_size=(target_size, target_size), # no target size as we are not downsampling.
    )
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # pad the x array dimension to conform to 4D tensor.

    # pass the list of multiple images np.vstack()
    images = np.vstack([x])
    # with CustomObjectScope({'BilinearUpsampling': BilinearUpsampling}):
    input_model = deeplabv3_plus(input_shape=(240, 320, 3), num_classes=1)
    input_model.load_weights(path_input_model_weight)
    output = input_model.predict(images)

    gray_scale_matrix = np.squeeze(output).astype(np.uint8)
    img = Image.fromarray(gray_scale_matrix)
    # Path(input_image).Parent.join("Output_" + Path(input_image).name)
    img.save(r"C:\Git\MarkerTrainer\data_test\result.jpg")

    print("COMPLETED")


def predict_folder(path_model_weights: str, input_folder: str):
    """
    Folder version of the prediction function provided above.
    :param input_model:
    :param input_folder:
    :param target_size:
    :return:
    """

    # Load model
    assert os.path.exists(path_model_weights)

    # Load files that predictions will be run upon.
    assert os.path.exists(input_folder)

    # with CustomObjectScope({'BilinearUpsampling': BilinearUpsampling}):
    input_model = deeplabv3_plus(input_shape=(240, 320, 3), num_classes=1)
    input_model.load_weights(path_model_weights)

    list_files = recursive_list(input_folder)

    for file in tqdm(list_files):

        # skip if it is not amn image.
        if "JPEG" not in file.upper() and "PNG" not in file.upper() and "JPEG" not in file.upper():
            continue

        # predicting multiple images at once
        img = image.load_img(file)

        x = image.img_to_array(img)
        x = np.expand_dims(
            x, axis=0
        )  # pad the x array dimension to conform to 4D tensor.

        # pass the list of multiple images np.vstack()
        images = np.vstack([x])
        output = input_model.predict(images)

        gray_scale_matrix = np.squeeze(output).astype(np.uint8)
        img = Image.fromarray(gray_scale_matrix)
        # Path(input_image).Parent.join("Output_" + Path(input_image).name)
        img.save(file + "_Output.jpg")

    write_JSON_records(path_model_weights, list_files, input_folder)

    print("Test folder prediction completed. ")

def write_JSON_records(path_model, list_images, destination):
    """
    Keep a simple JSON record of where the model came from
    :param path_model:
    :return:
    """
    data = {}
    data['model'] = path_model
    data['images'] = list_images
    path_json = Path(destination) / (unique_name() + "_prediction_details.json")
    write_json(path_json, data)
    print("JSON record written for the prediction process.")


if __name__ == "__main__":
    predict_folder(
        r"C:\Git\MarkerTrainer\models\2019-09-21T22_12_07.660353_Weights_model.Kaggle_DeepLabV3Plus.ModelClassSpec.h5",
        r"C:\Git\MarkerTrainer\data_test_results_2019-09-21T213456EST"
    )
