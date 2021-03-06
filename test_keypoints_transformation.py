import albumentations as A
import cv2


def read_n_parse_label(link, should_convert=False, img_shape=[0, 0]):
    """
    Read and convert image to regular coordinates
    :param link:
    :param should_convert:
    :param img_shape:
    :return:
    """
    line = open(link, "r").readlines()[0].split(",")
    coordinates = [float(e) for e in line[1:9]]
    height, width = img_shape[:2]

    if should_convert:
        return [
            [int(coordinates[0] * width), int(coordinates[4] * height)],
            [int(coordinates[1] * width), int(coordinates[5] * height)],
            [int(coordinates[2] * width), int(coordinates[6] * height)],
            [int(coordinates[3] * width), int(coordinates[7] * height)],
        ]

    return [
        [coordinates[0], coordinates[4]],
        [coordinates[1], coordinates[5]],
        [coordinates[2], coordinates[6]],
        [coordinates[3], coordinates[7]],
    ]


def read_img(link):
    """
    Read image using Opencv
    :param link:
    :return:
    """
    return cv2.imread(link)


def plot_img(img):
    """
    Plot image
    :param img:
    :return:
    """
    resized = cv2.resize(img, (512, 512))
    cv2.imshow("Image", resized)
    cv2.waitKey(-1)


def write_labels_to_img(img, points: list, is_normalize=True):
    """
    Add markers to image
    :param img:
    :param points:
    :param is_normalize:
    :return:
    """
    height, width = img.shape[:2]

    for point in points:
        if is_normalize:
            x = int(point[0] * width)
            y = int(point[1] * height)
        else:
            x = int(point[0])
            y = int(point[1])

        cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_SQUARE, 5)

    return img


def transform_img(img, labels):
    """
    Transform image
    :param img:
    :param labels:
    :return:
    """
    transform_func = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ], keypoint_params=A.KeypointParams(format='xy'))
    transformed = transform_func(image=img, keypoints=labels)

    return transformed["image"], transformed["keypoints"]


should_convert = True
img = read_img("raw-data/0000001.jpg")
labels = read_n_parse_label("raw-data/0000001.txt", should_convert, img.shape)

img, labels = transform_img(img, labels)
img = write_labels_to_img(img, labels, is_normalize=not should_convert)
plot_img(img)
