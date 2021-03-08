import albumentations as A
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np

from label import Label
from utils import IOU


class LicenseDataset(Dataset):
    def __init__(self, transforms: list, directory: str, input_size: int, output_size: int):
        self.img_links = glob.glob("{}/*.jpg".format(directory))
        self.transforms = A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy'))
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.img_links)

    def __getitem__(self, index):
        img_link = self.img_links[index]
        txt_link = img_link.split(".")[0] + ".txt"

        img = cv2.imread(img_link)
        label = Label.read_n_parse_label(txt_link, should_convert=True, img_shape=img.shape)

        transformed = self.transforms(image=img, keypoints=[
            label.tl(),
            label.tr(),
            label.br(),
            label.bl(),
        ])
        img = transformed["image"]
        keypoints = transformed["keypoints"]
        label = Label(keypoints[0], keypoints[1], keypoints[2], keypoints[3])
        # augment_sample --> apply albumentation
        # labels2output_map --> reimplement
        output = self.convert_labels_to_output(label)

        return img, output

    def convert_labels_to_output(self, label):
        """
        Transform image
        :param img:
        :param labels:
        :return:
        """
        output = np.zeros((self.output_size, self.output_size, 7), dtype=float)
        # resized_img = cv2.resize(img, (self.output_size, self.output_size))

        input_dim = np.array([self.input_size, self.input_size], dtype=float)
        output_dim = np.array([self.output_size, self.output_size], dtype=float)
        left, top = np.floor(label.tl() / input_dim * output_dim).astype(int).tolist()
        right, bottom = np.ceil(label.br() / input_dim * output_dim).astype(int).tolist()
        print(left, top, right, bottom)

        # compute IOU between center point and each index
        for y in range(top, bottom):
            for x in range(left, right):
                mn = np.array([x, y]) / output_dim * input_dim
                iou = IOU(mn - label.wh() / 2, mn + label.wh() / 2, label.tl(), label.br())

                if iou > 0.4:
                    p_WH = lppts * WH.reshape((2, 1))
                    p_MN = p_WH / stride

                    p_MN_center_mn = p_MN - mn.reshape((2, 1))

                    p_side = p_MN_center_mn / side

                    output[x, y, 0] = 1.
                    output[x, y, 1:] = p_side.T.flatten()
                # iou()

                # if iou > 0.5:
                #     output[x, y, 0] = 1.
                # calculate IOU between

        return output


if __name__ == "__main__":
    input_dim = 208
    output_dim = 13

    transforms = [
        # A.RandomCrop(512, 512, True),
        A.Resize(input_dim, input_dim),
        A.HorizontalFlip()
    ]

    dataset = LicenseDataset(transforms, "raw-data", input_dim, output_dim)
    # cv2.imshow("Test", dataset[2][1][:, :, 0])
    # cv2.waitKey(-1)

    for i in range(len(dataset)):
        dataset[i][1][:, :, 0]
        pass
        # cv2.imshow("Test", dataset[i][1][:, :, 0])
        # cv2.waitKey(-1)
