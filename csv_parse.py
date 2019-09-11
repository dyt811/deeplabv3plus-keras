# Here are some functions to help load the data set for the Severstal.
from pathlib import Path
import pandas as pd
import io
from PIL import ImageDraw, Image
from PythonUtils.folder import create
from collections import namedtuple
Pixel = namedtuple('Point', 'x y')

def parse_CSV(
        path_csv:Path = "../../data_1/train.csv",
        train_image_folder: Path = Path("../../data_1/train_images/"),
        output_image_folder: Path = Path("../../data_1/labelled_images/")):
    """
    A specialized function for Seversteel's CSV parsing.
    :param path_csv:
    :return:
    """
    create(output_image_folder)

    # Read Column 1: Get images and class ID separated.
    df = pd.read_csv(path_csv)

    # Preprocess to split the ImageId_ClassID into two columns and then add headers for them.
    df_new = df.ImageId_ClassId.str.split("_", expand=True)
    df_new.columns = ["Files", "DefectClass"]

    # Construct the final dataframe, which has headers: Files, DefectClass, and EncodedPixels
    df_final = pd.concat([df_new, df.EncodedPixels], axis=1)

    # Getting unique files names
    files = set(df_final["Files"])

    # Iterate through all files,
    for file in files:

        # Open each file,
        image = Image.open(train_image_folder.joinpath(Path(file)))
        draw = ImageDraw.Draw(image)

        defect_found = False
        # Iterate through the defects and check for them.
        for defect in range(1, 4):
            data_selected = df_final[df_final["DefectClass"] == f"{defect}"]
            data_selected = data_selected[data_selected["Files"] == f"{file}"]

            # Check if the EncodedPixels are empty,
            if pd.isna(data_selected.iloc[0, 2]):
                continue

            pixel_string = data_selected["EncodedPixels"].iat[0]
            list_coordinate_tuples = parse_coordinate(pixel_string)
            list_xy = decode_ListOrderLength_to_listxy(list_coordinate_tuples)
            # Draw the coordinates provided. for
            draw.point(list_xy, fill=(defect*40, defect*40, 0, 128))
            defect_found = True

        # After all four iteration, save the file.
        if defect_found:
            image.save(output_image_folder.joinpath(Path(file)))
        image.close()

    # Read Column 4: Writeout the data into another format.

def parse_coordinate(long_string: str, separator:str = " ") -> list:
    """
    A helper function to parse coordinate strings stored in CSVs
    :param long_string: input string.
    :param separator: the separator that
    :return:
    """
    #29102 12 29346 24 29602 24 29858 24 30114 24 30370 24 30626 24 30882 24 31139 23 31395 23 31651 23 31907 23 32163 23 32419 23 32675 23 77918 27 78174 55 78429 60 78685 64 78941 68 79197 72 79452 77 79708 81 79964 85 80220 89 80475 94 80731 98 80987 102 81242 105 81498 105 81754 104 82010 104 82265 105 82521 31 82556 69 82779 27 82818 63 83038 22 83080 57 83297 17 83342 50 83555 13 83604 44 83814 8 83866 37 84073 3 84128 31 84390 25 84652 18 84918 8 85239 10 85476 29 85714 47 85960 57 86216 57 86471 58 86727 58 86983 58 87238 59 87494 59 87750 59 88005 60 88261 60 88517 60 88772 61 89028 53 89283 40 89539 32 89667 10 89795 30 89923 28 90050 29 90179 37 90306 27 90434 38 90562 14 90690 38 90817 9 90946 38 91073 3 91202 38 91458 38 91714 38 91969 39 92225 39 92481 39 92737 39 92993 39 93248 40 93504 40 93760 40 94026 30 94302 10 189792 7 190034 21 190283 28 190539 28 190795 28 191051 28 191307 28 191563 28 191819 28 192075 28 192331 28 192587 28 192843 23 193099 14 193355 5

    list_coordinates_strips = [] # Note the coding style is coordinate + length, not x, y
    list_starting_pixel = []
    list_strip_length = []

    list_strings = long_string.split(separator)

    # if the list is odd, something went wrong.
    if len(list_strings) % 2 == 1:
        return list_coordinates_strips # return empty list if something wrong with the coordinate string

    # separate the list into two lists.
    for index, coordinate in enumerate(list_strings):
        if index % 2 == 0:
            list_starting_pixel.append(int(coordinate))
        else:
            list_strip_length.append(int(coordinate))

    # Zip into list of tuples.
    list_coordinates_strips = list(zip(list_starting_pixel, list_strip_length))
    return list_coordinates_strips


def decode_ListOrderLength_to_listxy(list_orderlength):
    """
    Decode a list of order length pixel style marker to the more human list of x y coordinate
    :param list_orderlength:a list of pixel order (n-th pixel) and length of the commit
    :return: list of tuple(x,y)
    """
    from itertools import chain
    list_xy = []
    for pair in list_orderlength:
        xy = decode_OrderLength_tuple_to_listxy(pair)
        list_xy = list_xy + xy
    return list_xy


def decode_OrderLength_tuple_to_listxy(pair: tuple):
    order = pair[0]  # commencing pixel
    length = pair[1]  # length

    pixel_start = decode_order_to_xy(order)
    list_pixels = decode_PixelLength_tuple_to_listxy(pixel_start, length)
    return list_pixels


def decode_PixelLength_tuple_to_listxy(coordinate: Pixel, length):
    """
    Specialized function to give the coordinate of the end pixel.

    eg. x1599, y2 + length 5, it should return
    x3, y3 properly. Inclusive of both end.

    x and y are ONE based, not zero index based.

    :param coordinate:
    :param length:
    :return:
    """
    list_pixels = []
    if (coordinate.x + length - 1) <= 1600:
        for xcoor in range(coordinate.x, (coordinate.x + length), 1):
            list_pixels.append((xcoor, coordinate.y))
    else:
        #(coordinate.x + length) > 1600:

        # For current row.
        for xcoor in range(coordinate.x, 1601, 1):
            list_pixels.append((xcoor, coordinate.y))
        remain_pixels = coordinate.x+length-1600
        # For next row.
        for xcoor in range(1, remain_pixels, 1):
            list_pixels.append((xcoor, coordinate.y+1))

    return list_pixels

def decode_order_to_xy(pixel_order:int)-> Pixel:
    """
    Take an order integer in a 1600 x 255 images, convert it to xy coordinate in the form of a Pixel object
    :param pixel_order: integer order of the pixel
    :return: Pixel object
    """
    y = pixel_order // 1600
    x = pixel_order % 1600
    return Pixel(x, y)

def test_parse_coordinate():
    list_orderlenth = parse_coordinate(
        "29102 12 29346 24 29602 24 29858 24 30114 24 30370 24 30626 24 30882 24 31139 23 31395 23 31651 23 31907 23 32163 23 32419 23 32675 23 77918 27 78174 55 78429 60 78685 64 78941 68 79197 72 79452 77 79708 81 79964 85 80220 89 80475 94 80731 98 80987 102 81242 105 81498 105 81754 104 82010 104 82265 105 82521 31 82556 69 82779 27 82818 63 83038 22 83080 57 83297 17 83342 50 83555 13 83604 44 83814 8 83866 37 84073 3 84128 31 84390 25 84652 18 84918 8 85239 10 85476 29 85714 47 85960 57 86216 57 86471 58 86727 58 86983 58 87238 59 87494 59 87750 59 88005 60 88261 60 88517 60 88772 61 89028 53 89283 40 89539 32 89667 10 89795 30 89923 28 90050 29 90179 37 90306 27 90434 38 90562 14 90690 38 90817 9 90946 38 91073 3 91202 38 91458 38 91714 38 91969 39 92225 39 92481 39 92737 39 92993 39 93248 40 93504 40 93760 40 94026 30 94302 10 189792 7 190034 21 190283 28 190539 28 190795 28 191051 28 191307 28 191563 28 191819 28 192075 28 192331 28 192587 28 192843 23 193099 14 193355 5")
    print(
        decode_ListOrderLength_to_listxy(list_orderlenth)
    )


def test_decode_order_to_xy():
    print(decode_order_to_xy(23))
    print(decode_order_to_xy(11123))

if __name__=="__main__":
    parse_CSV()
