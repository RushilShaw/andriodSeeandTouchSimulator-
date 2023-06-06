import cv2 as cv
import numpy as np
from pathlib import Path
from ppadb.client import Client
from sklearn.cluster import DBSCAN


def get_hu_device() -> Client.DEVICE:
    """
    This function requires ADB to be running or else a RuntimeError error will arise
    This function gets the hu_device and returns it. If the device is not found then it will return None
    :return: Client.DEVICE
    """
    client = Client()
    devices = client.devices()
    client.devices()
    hu_device = None
    for device in devices:
        if "Harman" in device.shell("getprop ro.product.manufacturer"):
            hu_device = device
            break
    return hu_device


def take_screenshot_of_android_device(hu_device: Client.DEVICE, screenshot_path: Path):
    """
    This function takes a screenshot of an android device to the path that is specified
    :param hu_device: a Client.DEVICE that is specified by the ppadb library
    :param screenshot_path: screenshot_path: pathlib Path object that defines the location where the path is saved
    :return: None
    """
    if hu_device is not None:
        result = hu_device.screencap()
        with open(f"{screenshot_path.resolve()}", "wb") as fp:
            fp.write(result)
        return hu_device


def find_image_locations(main_image_path: Path, sub_image_path: Path, confidence_interval: float) -> list:
    """
    Given a main image and a sub-image. This function will check the main image for instances of the sub-images
    If none are found then the functions will return an empty list
    If the paths of the images do not exist then the function will return a FileNotFound exception
    This function also removes similar values from a list
    It returns the center of the main image where the sub image was found

    :param main_image_path: a pathlib path object which stores the location of the main image
    :param sub_image_path: a pathlib path object which stores the location of the sub image
    :param confidence_interval: the tolerance at which the sub must fit into the main [0.0, 1.0] typically 0.7
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
        raise exception_group[0]
        # if len(exception_group) == 1:
        #     raise exception_group[0]
        # else:
        #     raise ExceptionGroup("Arguments to the Function were not Valid", exception_group)

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


def press_on_android_device(hu_device: Client.DEVICE, location_x: int, location_y: int):
    """
    This function will press on the x and y location of the client device
    :param hu_device: the device that is going to be pressed
    :param location_x: int - the x location where the device is pressed
    :param location_y: int - the y location where the device is pressed
    :return:
    """
    hu_device.shell(f"input tap {location_x} {location_y}")


def main():
    screenshot_path = Path("./screenshot.png")
    to_find_image_path = Path("./to_find.png")
    hu_device = get_hu_device()
    assert hu_device is not None, "The hu_device defined by the get_hu_device() function was not found"
    take_screenshot_of_android_device(hu_device, screenshot_path)
    found_points = find_image_locations(screenshot_path, to_find_image_path, confidence_interval=0.8)
    assert len(found_points) == 1, f"At least/Only 1 point should be found. However {len(found_points)} were found"
    to_press_location = found_points[0]
    press_on_android_device(hu_device, location_x=to_press_location[0], location_y=to_press_location[1])


if __name__ == '__main__':
    main()
