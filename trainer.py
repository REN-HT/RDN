import torch
import copy
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.dataSet import DataSet
from dataset.dataSet import ValidDataset
from model.rdn import RDNet
from config import opt
from torch.autograd import Variable
from model.rdn import L1_Charbonnier_loss
from utils.utils import denormalize, convert_rgb_to_y, calculate_psnr
import visdom


def train():
    # 训练集载入
    dataset = DataSet(opt.train_root)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker, drop_last=True)
    # 验证集载入
    validset =ValidDataset(opt.valid_root)
    valid_dataloader = DataLoader(validset, batch_size=1)

    net = RDNet()
    # static_dic = torch.load('D:/AProgram/SR/RDN/best_RDNet_weight.pth')
    # net.load_state_dict(static_dic)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=1e-4)
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = L1_Charbonnier_loss()
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    if opt.cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    best_weight = copy.deepcopy(net.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    vis = visdom.Visdom(env=u'rdn')

    for epoch in range(opt.epoch):
        net.train()
        # 训练
        train_loss = 0
        for ii, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            lr, target = Variable(data[0]), Variable(data[1])
            if opt.cuda:
                lr = lr.cuda()
                target = target.cuda()

            output = net(lr)
            loss = criterion(output, target)
            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
            # lr_scheduler.step()
            train_loss += loss.item()
            print('{} epoch loss:{:.3f}'.format(epoch+1, train_loss/(ii+1)))
        vis.line(X=torch.Tensor([epoch+1]), Y=torch.Tensor([train_loss/len(dataloader)]), win='loss', update='append', opts=opt.opts1)

        if (epoch+1) % 5 == 0:
            torch.save(net.state_dict(), '2xRDNet_weight{}.pth'.format((epoch+1) / 5))

        # 验证
        net.eval()
        epoch_psnr = 0

        for valid_data in valid_dataloader:
            inputs, labels = valid_data
            if opt.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # 无梯度计算，不参与更新
            with torch.no_grad():
                preds = net(inputs)
            # RGB转Y通道计算
            preds = convert_rgb_to_y(denormalize(preds.squeeze(0)))
            labels = convert_rgb_to_y(denormalize(labels.squeeze(0)))

            preds = preds[opt.scale:-opt.scale, opt.scale:-opt.scale]
            labels = labels[opt.scale:-opt.scale, opt.scale:-opt.scale]

            # 计算峰值信噪比
            epoch_psnr += calculate_psnr(preds, labels)
        mean_psnr = epoch_psnr/len(valid_dataloader)
        print('eval psnr:{:.3f}'.format(mean_psnr))
        vis.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([mean_psnr]), win='psnr', update='append', opts=opt.opts2)

        # 记录最优迭代
        if best_psnr < mean_psnr:
            best_epoch = epoch
            best_psnr = mean_psnr
            best_weight = copy.deepcopy(net.state_dict())
    print('best_epoch {}, best_psnr {:.3f}'.format(best_epoch, best_psnr))
    torch.save(best_weight, 'best_2xRDNet_weight.pth')


if __name__ == '__main__':
    train()
