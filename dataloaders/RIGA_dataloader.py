import math

from torch.utils import data
import numpy as np
from PIL import Image, ImageFilter
from batchgenerators.utilities.file_and_folder_operations import *
EPS = 1e-7
def center_coordinates(binary_segmentation):
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool_)

    vertical_axis_diameter = np.sum(binary_segmentation, axis=0)
    y = np.argmax(vertical_axis_diameter)

    Horizontal_axis_diameter = np.sum(binary_segmentation, axis=1)
    x = np.argmax(Horizontal_axis_diameter)

    return x, y

def calculateCompact(image):
    #image = Image.open(image).convert('L')
    image = np.asarray(image, np.float32)
    image_od = np.zeros(image.shape, dtype=np.float32)
    image_oc = np.zeros(image.shape, dtype=np.float32)

    image_od[image == 255] = 255
    image_od[image == 128] = 255
    image_oc[image == 128] = 255

    image_od = Image.fromarray(np.uint8(image_od))
    image_oc = Image.fromarray(np.uint8(image_oc))
    edge_od = image_od.filter(ImageFilter.FIND_EDGES)
    edge_oc = image_oc.filter(ImageFilter.FIND_EDGES)

    edge_od = np.asarray(edge_od, np.float32)
    edge_oc = np.asarray(edge_oc, np.float32)

    image_od = np.asarray(image_od, np.float32)
    image_oc = np.asarray(image_oc, np.float32)

    image_od = image_od / 255
    image_oc = image_oc / 255

    edge_od = edge_od / 255
    edge_oc = edge_oc / 255
    # 中心坐标
    center_od_x, center_od_y = center_coordinates(image_od > 0)
    center_oc_x, center_oc_y = center_coordinates(image_oc > 0)
    # 边界坐标
    edge_od_arg_x, edge_od_arg_y= np.where(edge_od == 1)
    edge_oc_arg_x, edge_oc_arg_y = np.where(edge_oc == 1)

    d_j_od = pow(pow(edge_od_arg_x - center_od_x, 2) + pow(edge_od_arg_y - center_od_y, 2), 0.5)
    d_mean_od = np.mean(d_j_od)
    v_od = pow(np.mean(pow(d_j_od - d_mean_od, 2))/(d_j_od.shape[0]+EPS), 0.5)
    d_j_oc = pow(pow(edge_oc_arg_x - center_oc_x, 2) + pow(edge_oc_arg_y - center_oc_y, 2), 0.5)
    d_mean_oc = np.mean(d_j_oc)
    v_oc = pow(np.mean(pow(d_j_oc - d_mean_oc, 2)) / (d_j_oc.shape[0] + EPS), 0.5)
    # 周长
    p_od = np.sum(edge_od)
    p_oc = np.sum(edge_oc)
    # 面积
    a_od = np.sum(image_od)
    a_oc = np.sum(image_oc)

    c_od = (4*math.pi*a_od)/pow(p_od, 2)
    c_oc = (4*math.pi*a_oc)/pow(p_oc, 2)

    return [np.nan_to_num(c_od/(v_od+EPS)), np.nan_to_num(c_oc/(v_oc+EPS))]

class RIGA_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=(512, 512), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        label_file = join(self.root, self.label_list[item])
        img = Image.open(img_file)
        label = Image.open(label_file)
        img = img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        weight = calculateCompact(label)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        label_npy = np.array(label)
        mask = np.zeros_like(label_npy)
        mask[label_npy > 0] = 1
        mask[label_npy == 128] = 2
        return img_npy, mask[np.newaxis], img_file, weight


class RIGA_unlabeled_set(data.Dataset):
    def __init__(self, root, img_list, target_size=(512, 512), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        img = Image.open(img_file)
        img = img.resize(self.target_size)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        return img_npy, None, img_file
