import PIL.Image
from PIL import Image
from torchvision import transforms as T
from model.rdn import RDNet
import torch
import os
import numpy as np
from utils.utils import denormalize, convert_rgb_to_y, calculate_psnr
from config import opt


def resize(img, h, w):
    trans = T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC)
    return trans(img)


def real_test(path):
    img = Image.open(path).convert('RGB')
    # w, h = img.size
    # y, b, r = img.split()
    # b = resize(b, h*4, w*4)
    # r = resize(r, h*4, w*4)
    input = T.ToTensor()(img).unsqueeze(0)
    model = RDNet()
    model_state_dic = torch.load('D:/AProgram/SR/RDN/best_2xRDNet_weight.pth')
    model.load_state_dict(model_state_dic)
    if opt.cuda:
        input = input.cuda()
        model = model.cuda()
    with torch.no_grad():
        out4x = model(input).clamp(0.0, 1.0)
    out4x = out4x.squeeze(0)
    out4x = T.ToPILImage()(out4x)
    out4x.show()
    # imgout = Image.merge('YCbCr', (out4x, b, r))


def test(path, scale):
    gt = Image.open(path).convert('RGB')
    w, h = gt.size
    w, h = (w // scale) * scale, (h // scale) * scale
    img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
    lr = img.resize((w // scale, h // scale), resample=PIL.Image.BICUBIC)
    input = T.ToTensor()(lr).unsqueeze(0)
    model = RDNet()
    model_state_dic = torch.load('D:/AProgram/SR/RDN/best_2xRDNet_weight.pth')
    model.load_state_dict(model_state_dic)
    if opt.cuda:
        input = input.cuda()
        model = model.cuda()
    with torch.no_grad():
        out = model(input).clamp(0.0, 1.0)
    out = out.squeeze(0)
    out = T.ToPILImage()(out)
    out.show()


# 计算峰值信噪比
def PSNRRGB(root, scale):
    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    to_tensor = T.ToTensor()
    net = RDNet()
    model_state_dic = torch.load('D:/AProgram/SR/RDN/best_2xRDNet_weight.pth')
    net.load_state_dict(model_state_dic)
    res = 0
    for path in img_paths:
        gt = Image.open(path).convert('RGB')
        w, h = gt.size
        w, h = (w // scale) * scale, (h // scale) * scale
        img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
        lr = img.resize((w//scale, h//scale), resample=PIL.Image.BICUBIC)
        lr = to_tensor(lr).unsqueeze(0)
        if opt.cuda:
            lr = lr.cuda()
            net = net.cuda()
        with torch.no_grad():
            preds = net(lr).squeeze(0)
        labels = to_tensor(img)

        preds = convert_rgb_to_y(denormalize(preds.cpu()))
        labels = convert_rgb_to_y(denormalize(labels))
        preds = preds[scale:-scale, scale:-scale]
        labels = labels[scale:-scale, scale:-scale]

        res += calculate_psnr(preds, labels)
    print('PSNR:', res/len(img_paths))


def SSIM(root, scale):
    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    to_tensor = T.ToTensor()
    net = RDNet()
    model_state_dic = torch.load('D:/AProgram/SR/RDN/2xRDNet_weight40.0.pth')
    net.load_state_dict(model_state_dic)
    res = 0
    for path in img_paths:
        gt = Image.open(path).convert('RGB')
        w, h = gt.size
        w, h = (w // scale) * scale, (h // scale) * scale
        img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
        lr = img.resize((w//scale, h//scale), resample=PIL.Image.BICUBIC)
        input = to_tensor(lr).unsqueeze(0)
        if opt.cuda:
            input = input.cuda()
            net = net.cuda()
        with torch.no_grad():
            preds = net(input).squeeze(0)
        labels = to_tensor(img)

        preds = convert_rgb_to_y(denormalize(preds.cpu()))
        labels = convert_rgb_to_y(denormalize(labels))

        res += calculate_SSIM(preds, labels)

    print('SSIM:', res / len(img_paths))


# 计算两幅图片结构相似比
def calculate_SSIM(sr, hr):
    assert len(sr.shape) == len(hr.shape)
    assert sr.shape == hr.shape
    # sr,hr均值
    u1 = sr.mean()
    u2 = hr.mean()
    # sr,hr标准差
    sigma1 = torch.sqrt(((sr-u1)**2).mean())
    sigma2 = torch.sqrt(((hr-u2)**2).mean())
    # sr,hr协方差
    covariance = ((sr-u1)*(hr-u2)).mean()
    # 固定系数,1.0为最大像数值，由于这里为Tensor类型,所以最大为1，如果像素值范围是【0-255】则为255
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    return ((2*u1*u2+c1)*(2*covariance+c2))/((u1**2+u2**2+c1)*(sigma1**2+sigma2**2+c2))


if __name__=='__main__':
    path = 'F:/dataset/SuperResolutionDataset/Test/Mix/baby_GT.bmp'
    path1 = 'F:/dataset/SuperResolutionDataset/Test/Set5'
    path2 = 'F:/dataset/SuperResolutionDataset/Test/Set4x'
    # test(path)
    PSNRRGB(path1, 2)
    # SSIM(path1, 2)

