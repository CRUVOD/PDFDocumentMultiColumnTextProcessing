import numpy as np

from VieTableRecognition.ImageProcessing import *

"""Lattice method of parsing looks for lines between text
to parse the table.

Give an image in the form of a np.ndarray, return table bounding box in image

"""

def Generate_Table_bbox(image:np.ndarray):
    image, thresholdedImage = adaptive_threshold(image)

    regions = None

    vertical_mask, vertical_segments = find_lines(
        thresholdedImage,
        regions=regions,
        direction="vertical",
    )
    horizontal_mask, horizontal_segments = find_lines(
        thresholdedImage,
        regions=regions,
        direction="horizontal",
    )

    contours = find_contours(vertical_mask, horizontal_mask)
    table_bbox = find_joints(contours, vertical_mask, horizontal_mask)

    return table_bbox

def Extract_Table_Meta_Info(image):
    """
    Returns the bounding box of a table in an image
    """

    return Generate_Table_bbox(image)