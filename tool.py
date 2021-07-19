import PIL.Image
import cv2
import os
import h5py
from PIL import Image
from torchvision import transforms as T


# 用于裁剪图片
def clip_image(root):
    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    count = 1
    for path in img_paths:
        im = cv2.imread(path)
        row, col, _ = im.shape
        for i in range(0, row, 200):
            for j in range(0, col, 200):
                if i+480 >= row or j+480 >= col:
                    continue
                imm = im[i:i+480, j:j+480]
                img_path = 'F:/dataset/SuperResolutionDataset/480x480DIV2K_train_HR'
                save_path = os.path.join(img_path, str(count)+'.png')
                cv2.imwrite(save_path, imm)
                count += 1


# 计算训练集均值和标准差,单通道
def calculate_mean_std(root):
    img_names = os.listdir(root)
    to_tensor=T.ToTensor()
    img_paths=[os.path.join(root, name) for name in img_names]
    r_mean=0;g_mean=0;b_mean=0
    r_std=0;g_std=0;b_std=0
    for path in img_paths:
        img=Image.open(path).convert('RGB')
        r, g, b = img.split()
        r = to_tensor(r)
        g = to_tensor(g)
        b = to_tensor(b)
        r_mean+=r.mean(); g_mean+=g.mean(); b_mean+=b.mean()
        r_std+=r.std(); g_std+=g.std(); b_std+=b.std()
    print("mean:{},{},{}".format(r_mean/len(img_paths), g_mean/len(img_paths), b_mean/len(img_paths)))
    print("std:{},{},{}".format(r_std / len(img_paths), g_std / len(img_paths), b_std / len(img_paths)))


# 创建h5_file文件
def create_h5_file(root, scale):
    h5_file = h5py.File('F:/dataset/SuperResolutionDataset/Train/2x_div2k_file.h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    index = 0

    for img_path in img_paths:
        hr = Image.open(img_path).convert('RGB')
        lr = hr.resize((hr.width//scale, hr.height//scale), resample=PIL.Image.BICUBIC)
        hr = T.ToTensor()(hr)
        lr = T.ToTensor()(lr)
        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)
        index += 1

    h5_file.close()


def create_h5_file_valid(root, scale):
    h5_file = h5py.File('F:/dataset/SuperResolutionDataset/Train/2x_div2k_file_valid.h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    valid_img_names = os.listdir(root)
    paths = [os.path.join(root, name) for name in valid_img_names]
    pos = 0

    for path in paths:
        hr = Image.open(path).convert('RGB')
        for hr in T.FiveCrop(size=(hr.height//2, hr.width//2))(hr):
            hr = hr.resize(((hr.width//scale)*scale, (hr.height//scale)*scale), resample=PIL.Image.BICUBIC)
            lr = hr.resize((hr.width//scale, hr.height//scale), resample=PIL.Image.BICUBIC)

            hr = T.ToTensor()(hr)
            lr = T.ToTensor()(lr)

            lr_group.create_dataset(str(pos), data=lr)
            hr_group.create_dataset(str(pos), data=hr)
            pos += 1

    h5_file.close()

if __name__=='__main__':
    path = 'F:/dataset/SuperResolutionDataset/valid/temp'
    path1 = 'F:/dataset/SuperResolutionDataset/480x480DIV2K_train_HR'
    create_h5_file_valid(path, 2)
