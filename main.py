import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN


def find_image_locations(main_image_path: Path, sub_image_path: Path, confidence_interval: float) -> list:
    """
    Given a main image and a sub-image. This function will check the main image for instances of the sub-images
    If none are found then the functions will return an empty list
    If the paths of the images do not exist then the function will return a FileNotFound exception
    This function also removes similar values from a list
    It returns the center of the main image where the sub image was found

    :param main_image_path: a pathlib path object which stores the location of the main image
    :param sub_image_path: a pathlib path object which stores the location of the sub image
    :param confidence_interval: the tolerance at which the sub must fit into the main (0.0, 1.0] typically 0.7
    :return: returns a list of tuples of 2d points that represent coordinates of the pixel location
    """
    # input validation
    exception_group = []
    if not main_image_path.exists():
        exception_group.append(FileNotFoundError("Main Image Path Not Found!"))
    if not sub_image_path.exists():
        exception_group.append(FileNotFoundError("Sub Image Path Not Found!"))
    if not (0 <= confidence_interval <= 1):
        exception_group.append(ValueError("Confidence Interval must be between 0 and 1"))
    if exception_group:
        raise ExceptionGroup("Arguments to the Function were not Valid", exception_group)

    # finds the location of all points where the sub image can be found in the main image
    main_image_rgb = cv.imread(f"{main_image_path.resolve()}")
    main_image_gray = cv.cvtColor(main_image_rgb, cv.COLOR_BGR2GRAY)
    sub_image = cv.imread(f"{sub_image_path.resolve()}", cv.IMREAD_GRAYSCALE)
    sub_image_width, sub_image_height = sub_image.shape[::-1]
    results = cv.matchTemplate(main_image_gray, sub_image, cv.TM_CCOEFF_NORMED)
    locations = np.where(results >= confidence_interval)

    # converts the points to return the center location where the image is found
    points = [(x_cord + sub_image_width // 2, y_cord + sub_image_height // 2)
              for x_cord, y_cord in zip(*locations[::-1])]

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    # uses the DBSCAN algorithm to remove any duplicate similar points
    cluster = DBSCAN(eps=5, min_samples=1).fit(np.array(points, dtype=int))
    filtered_points = []
    used_labels_set = set()
    for index, label in enumerate(cluster.labels_):
        if label not in used_labels_set:
            filtered_points.append(points[index])
            used_labels_set.add(label)
    return filtered_points


def main():
    main_image_path = Path("./large.png")
    sub_image_path = Path("./small.png")
    found_points = find_image_locations(main_image_path, sub_image_path, confidence_interval=0.8)
    assert len(found_points) == 1, f"At least/Only 1 point should be found. However {len(found_points)} were found"



if __name__ == '__main__':
    main()
