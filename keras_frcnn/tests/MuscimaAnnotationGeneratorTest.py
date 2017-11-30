import shutil
import unittest
from typing import List, Tuple

import os

from keras_frcnn.muscima_annotation_generator import create_annotations_in_plain_format, \
    create_annotations_in_pascal_voc_format


class MuscimaAnnotationGeneratorTest(unittest.TestCase):
    def test_create_annotations_in_plain_format(self):
        # Arrange
        objects_appearing_in_cropped_image = self.get_fake_annotations()
        annotation_file = "test_annotations.txt"
        self.delete_annotations_if_exist(annotation_file)

        # Act
        create_annotations_in_plain_format(annotation_file, objects_appearing_in_cropped_image)

        # Assert
        self.assertTrue(os.path.exists(annotation_file))
        with open(annotation_file, "r") as file:
            first_line = file.readline()
            self.assertEqual("file1.jpg,5,5,10,10,class1\n", first_line)

    def test_create_annotations_in_pascal_voc_format_expect_files_to_be_generated(self):
        # Arrange
        objects_appearing_in_cropped_image = self.get_fake_annotations()
        annotations_folder = "Test-Annotations"
        shutil.rmtree(annotations_folder, ignore_errors=True)

        # Act
        create_annotations_in_pascal_voc_format(annotations_folder, objects_appearing_in_cropped_image, 100, 100, 3)

        # Assert
        self.assertTrue(os.path.exists(annotations_folder))
        number_of_generated_annotation_files = len(os.listdir(annotations_folder))
        self.assertEqual(number_of_generated_annotation_files, 2, "Expecting 1 file per image file")

    def get_fake_annotations(self):
        objects_appearing_in_cropped_image: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
        objects_appearing_in_cropped_image.append(("file1.jpg", "class1", [5, 5, 10, 10]))
        objects_appearing_in_cropped_image.append(("file2.jpg", "class1", [12, 13, 14, 16]))
        objects_appearing_in_cropped_image.append(("file2.jpg", "class2", [25, 25, 20, 20]))
        return objects_appearing_in_cropped_image

    def test_create_annotations_in_pascal_voc_format_expect_files_to_contain_annotations(self):
        # Arrange
        objects_appearing_in_cropped_image = self.get_file2_fake_annotations()
        annotations_folder = "Test-Annotations"
        shutil.rmtree(annotations_folder, ignore_errors=True)

        # Act
        create_annotations_in_pascal_voc_format(annotations_folder, objects_appearing_in_cropped_image, 150, 200, 3)

        # Assert
        self.assertTrue(os.path.exists(annotations_folder))
        number_of_generated_annotation_files = len(os.listdir(annotations_folder))
        self.assertEqual(number_of_generated_annotation_files, 1, "Expecting 1 file per image file")
        with open("expected_annotation_results.xml", "r") as expected_file:
            expected_xml_document = expected_file.read()
        with open("Test-Annotations/file2.xml", "r") as actual_file:
            actual_xml_document = actual_file.read()

        self.assertEqual(expected_xml_document, actual_xml_document)

    def get_file2_fake_annotations(self):
        objects_appearing_in_cropped_image: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
        objects_appearing_in_cropped_image.append(("file2.jpg", "class1", [12, 13, 14, 16]))
        objects_appearing_in_cropped_image.append(("file2.jpg", "class2", [25, 25, 20, 20]))
        return objects_appearing_in_cropped_image

    def delete_annotations_if_exist(self, annotation_file):
        if (os.path.exists(annotation_file)):
            os.remove(annotation_file)


if __name__ == '__main__':
    unittest.main()
