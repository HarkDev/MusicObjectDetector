import os
from itertools import groupby
from lxml import etree
from typing import List, Tuple

from lxml.etree import Element, SubElement


def create_annotations_in_plain_format(exported_annotations_file_path: str, objects_appearing_in_cropped_image: List[
    Tuple[str, str, Tuple[int, int, int, int]]]):
    with open(exported_annotations_file_path, "a") as annotations_file:
        for object_appearing_in_cropped_image in objects_appearing_in_cropped_image:
            file_name = object_appearing_in_cropped_image[0]
            class_name = object_appearing_in_cropped_image[1]
            translated_bounding_box = object_appearing_in_cropped_image[2]
            trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
            annotations_file.write("{0},{1},{2},{3},{4},{5}\n".format(file_name,
                                                                      trans_left,
                                                                      trans_top,
                                                                      trans_right,
                                                                      trans_bottom,
                                                                      class_name))


def create_annotations_in_pascal_voc_format(annotations_folder: str,
                                            objects_appearing_in_cropped_image: List[
                                                Tuple[str, str, Tuple[int, int, int, int]]],
                                            image_width: int,
                                            image_height: int,
                                            image_depth: int):
    os.makedirs(annotations_folder, exist_ok=True)

    groups = [list(it) for k, it in
              groupby(objects_appearing_in_cropped_image, lambda annotation_tuple: annotation_tuple[0])]

    for group in groups:
        if len(group) == 0:
            continue

        file_name = group[0][0]

        annotation = Element("annotation")
        folder = SubElement(annotation, "folder")
        folder.text = "muscima_pp_cropped_images"
        filename = SubElement(annotation, "filename")
        filename.text = file_name
        source = SubElement(annotation, "source")
        database = SubElement(source, "database")
        database.text = "MUSCIMA++"
        source_annotation = SubElement(source, "annotation")
        source_annotation.text = "MUSCIMA++ (v0.9.1)"
        image = SubElement(source, "image")
        image.text = "CVC-MUSCIMA"
        size = SubElement(annotation, "size")
        width = SubElement(size, "width")
        width.text = str(image_width)
        height = SubElement(size, "height")
        height.text = str(image_height)
        depth = SubElement(size, "depth")
        depth.text = str(image_depth)

        # Write results to file
        for detected_object in group:
            class_name = detected_object[1]
            translated_bounding_box = detected_object[2]
            ymin, xmin, ymax, xmax = translated_bounding_box

            object = SubElement(annotation, "object")
            name = SubElement(object, "name")
            name.text = class_name
            pose = SubElement(object, "pose")
            pose.text = "Unspecified"
            truncated = SubElement(object, "truncated")
            truncated.text = "0"
            difficult = SubElement(object, "difficult")
            difficult.text = "0"
            bndbox = SubElement(object, "bndbox")
            bb_xmin = SubElement(bndbox, "xmin")
            bb_xmin.text = str(xmin)
            bb_ymin = SubElement(bndbox, "ymin")
            bb_ymin.text = str(ymin)
            bb_xmax = SubElement(bndbox, "xmax")
            bb_xmax.text = str(xmax)
            bb_ymax = SubElement(bndbox, "ymax")
            bb_ymax.text = str(ymax)

        xml_file_path = os.path.join(annotations_folder, os.path.splitext(file_name)[0] + ".xml")
        pretty_xml_string = etree.tostring(annotation, pretty_print=True)

        with open(xml_file_path, "wb") as xml_file:
            xml_file.write(pretty_xml_string)
