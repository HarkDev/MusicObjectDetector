import os
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def get_data(dataset_directory: str, annotation_file: str, visualise: bool = False) -> \
        Tuple[List[dict], List[dict], List[dict], dict, dict]:
    """

    :param dataset_directory: The directory that contains one folder for training/validation/test images
    :param annotation_file: The Annotations.txt file containing the bounding-boxes for all images
    :param visualise: If true, every image will be displayed with every bounding-box
    :return: Three dictionaries for training/validation/test images, a dictionary with the number of elements per class
             and a dictionary with a class mapping between class-id and class name.
    """
    found_bg = False
    classes_count = {}
    class_mapping = {}
    training_images = os.listdir(os.path.join(dataset_directory, "training"))
    validation_images = os.listdir(os.path.join(dataset_directory, "validation"))
    test_images = os.listdir(os.path.join(dataset_directory, "test"))
    training_data = {}
    validation_data = {}
    test_data = {}

    num_lines = sum(1 for line in open(annotation_file, 'r'))

    with open(annotation_file, 'r') as f:
        for line in tqdm(f, desc="Parsing annotation file", total=num_lines):
            line_split = line.strip().split(',')
            (filename, left, top, right, bottom, class_name) = line_split
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print("Found class name with special name bg. Will be treated as a background region (this is "
                          "usually for hard negative mining).")
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename in training_images:
                dataset_split = "training"
                filename = add_image_to_dictionary(class_name, dataset_directory, filename, training_data, left, right,
                                                   top, bottom, dataset_split)
            elif filename in validation_images:
                dataset_split = "validation"
                filename = add_image_to_dictionary(class_name, dataset_directory, filename, validation_data, left,
                                                   right, top, bottom, dataset_split)
            elif filename in test_images:
                dataset_split = "test"
                filename = add_image_to_dictionary(class_name, dataset_directory, filename, test_data, left, right,
                                                   top, bottom, dataset_split)

            if visualise:
                img = cv2.imread(filename)
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255))
                cv2.imshow('img', img)
                cv2.waitKey(0)

        all_training_data = convert_to_list(training_data)
        all_validation_data = convert_to_list(validation_data)
        all_test_data = convert_to_list(test_data)

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_training_data, all_validation_data, all_test_data, classes_count, class_mapping


def convert_to_list(data_dictionary):
    all_data = []
    for key in data_dictionary:
        all_data.append(data_dictionary[key])

    return all_data


def add_image_to_dictionary(class_name, dataset_directory, filename, data_dictionary, left, right, top, bottom,
                            dataset_split):
    filename = os.path.join(dataset_directory, dataset_split, filename)
    if filename not in data_dictionary:
        data_dictionary[filename] = {}

        img = cv2.imread(filename)
        (rows, cols) = img.shape[:2]
        data_dictionary[filename]['filepath'] = filename
        data_dictionary[filename]['width'] = cols
        data_dictionary[filename]['height'] = rows
        data_dictionary[filename]['bboxes'] = []
        data_dictionary[filename]['imageset'] = dataset_split

    data_dictionary[filename]['bboxes'].append(
        {'class': class_name, 'x1': left, 'x2': right, 'y1': top, 'y2': bottom})

    return filename


if __name__ == "__main__":
    training_images, validation_images, test_images, classes_count, class_mapping = get_data(
        "../data",
        "../data/Annotations.txt",
        False)

    number_of_bounding_boxes = sum(classes_count.values())
    print("Found {0} training, {1} validation and {2} test samples with {3} bounding-boxes belonging to {4} classes"
          .format(len(training_images), len(validation_images), len(test_images), number_of_bounding_boxes,
                  len(classes_count)))
