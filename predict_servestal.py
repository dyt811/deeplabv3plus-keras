from keras.preprocessing import image
import os
import numpy as np


from PythonUtils.PUFolder import recursive_list, recursive_list_re
from PythonUtils.PUFile import unique_name
from PythonUtils.PUJson import write_json
from PythonUtils.rle_encoding import RLE_encoding
from model.Kaggle_DeepLabV3Plus.ModelLayersSpec import deeplabv3_plus
from PIL import Image
import csv
from tqdm import tqdm
from pathlib import Path

shape_images = (256, 1600, 3)
classes = 4

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
    input_model = deeplabv3_plus(input_shape=shape_images, num_classes=classes)
    input_model.load_weights(path_input_model_weight)
    output = input_model.predict(images)

    gray_scale_matrix = np.squeeze(output).astype(np.uint8)
    img = Image.fromarray(gray_scale_matrix)
    # Path(input_image).Parent.join("Output_" + Path(input_image).name)
    img.save(r"C:\Git\MarkerTrainer\data_test\result.jpg")

    print("COMPLETED")


def predict_folder(path_model_weights: str, input_folder: str, path_output: Path=Path(r"C:\Git\MarkerTrainer\data_servestal\test_out")):
    """
    Folder version of the prediction function provided above, however, it also convert the results to CSV.
    :param input_model: the model to be used for prediction
    :param input_folder: input folder full of Servestal images.
    Will writeoutput to the folder with the POST FIX OUTPUT
    """

    # Load model
    assert os.path.exists(path_model_weights)

    # Load files that predictions will be run upon.
    assert os.path.exists(input_folder)

    input_model = deeplabv3_plus(input_shape=shape_images, num_classes=classes)
    input_model.load_weights(path_model_weights)

    list_files = recursive_list(input_folder)

    for file in tqdm(list_files):

        # skip if it is not amn image.
        if (
            "JPEG" not in file.upper()
            and "PNG" not in file.upper()
            and "JPG" not in file.upper()
        ):
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

        # Saving as NPY.
        np.save(path_output / (Path(file).name + ".npy"), gray_scale_matrix)


        """ Saveing as JPG 
        for defect_class in range(4):
            img = Image.fromarray(gray_scale_matrix[:, :, defect_class])
            # Path(input_image).Parent.join("Output_" + Path(input_image).name)
            img.save(file + f"_{defect_class}.jpg")
        """

    write_JSON_records(path_model_weights, list_files, input_folder)

    print("Test folder prediction completed. ")

def predict_CSVs(path_NPYs: Path, path_csv: Path):
    """
    Predict all the NPYs in a folder, into csv output at a location
    :param path_NPYs:
    :param path_csv:
    :return:
    """
    assert path_NPYs.is_dir()
    files = recursive_list_re(path_NPYs, 'npy')
    for file in tqdm(files):
        predict_CSV(Path(file), path_csv)

def predict_CSV(path_NPY: Path, path_csv: Path):
    """
    A wrapper after predict folder to convert  the results NPY to RLE.
    Take one NPY of ground truth and output into a row for path_CSV
    :return:
    """
    # Load NPY
    ground_truth = np.load(path_NPY)

    assert path_NPY.suffix == ".npy"
    name_image_file = path_NPY.stem
    #name_file = Path(name_image_file).stem

    # Make sure there are 4 layers, one for each defect.
    assert ground_truth.shape[2] == 4

    # Iterate through the layers.
    for class_defect in range(4):

        mask = ground_truth[:, :, class_defect]
        # Convert the data to RLE
        defect_encoding = RLE_encoding(mask)
        list_order_length = defect_encoding.get()
        tuple_order_length = sum(list_order_length, ())
        list_RLE = list(tuple_order_length)

        str_RLE = " ".join(repr(e) for e in list_RLE)

        # Remove comma from string
        #str_RLE = str_RLE.replace(",", "")

        file_exists = path_csv.exists()
        # Open file
        with open(path_csv, "a", newline="") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file, fieldnames=["ImageId_ClassId", "EncodedPixels"]
            )
            # Write header if it does not exist.
            if not file_exists:
                csv_writer.writeheader()
            # Write file/class ID and then the list of EncodedPixels
            csv_writer.writerow({"ImageId_ClassId": name_image_file+"_"+str(class_defect+1),
                                 "EncodedPixels": str_RLE})

def write_JSON_records(path_model, list_images, destination):
    """
    Keep a simple JSON record of where the model came from
    :param path_model:
    :return:
    """
    data = {}
    data["model"] = path_model
    data["images"] = list_images
    path_json = Path(destination) / (unique_name() + "_prediction_details.json")
    write_json(path_json, data)
    print("JSON record written for the prediction process.")


if __name__ == "__main__":

    predict_CSVs(
        Path(r"C:\Git\MarkerTrainer\data_servestal\test_out"),
        Path(r"C:\Git\MarkerTrainer\data_servestal\test_out\output.csv")
    )
    """
    predict_folder(
        path_model_weights=r"C:\Git\MarkerTrainer\models\2019-10-02T22_14_23.194370_FinalModelWeights_model.Kaggle_DeepLabV3Plus.ModelClassSpec.h5",
        input_folder=r"C:\Git\MarkerTrainer\data_servestal\test_images")
    """