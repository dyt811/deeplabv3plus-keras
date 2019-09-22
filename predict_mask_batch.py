import os
from pathlib import Path
from typing import List

from tqdm import tqdm
from PythonUtils.PUFolder import recursive_list
from PUFolder import create, flatcopy
from model.Kaggle_DeepLabV3Plus.predict_mask import predict_folder


def predict_folder_batch(path_models: List[str], path_input: str, path_output: Path):
    """
    A batch function which predicts using many models provided over input.
    :param path_models:
    :param path_input:
    :return:
    """

    for specific_model in tqdm(path_models):
        # Skip model if WEIGHT is not in the file name.
        if not "WEIGHT" in specific_model.upper():
            continue

        # Pathout + Timestamp + ModelFileName.
        path_output_model = path_output / os.path.basename(specific_model)
        create(path_output_model)
        # Copy all test data to the destination.
        flatcopy(path_input, path_output_model)

        # Run the prediction of the model against content from that folder.
        predict_folder(specific_model, str(path_output_model))

    print("Batch modelS predictions completed!")

if __name__=="__main__":
    models = recursive_list(r"C:\BatchModelTesting\models")
    path_output = Path(r"C:\BatchModelTesting")
    predict_folder_batch(models, r"C:\BatchModelTesting\test_sets", path_output)