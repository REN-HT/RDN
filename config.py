class DefaultConfig(object):
    train_root = 'D:/AProgram/data/2x_div2k_file.h5'
    valid_root = 'D:/AProgram/data/2x_div2k_file_valid.h5'
    batch_size = 16
    lr = 1e-4
    num_worker = 10
    epoch = 200
    cuda = True

    scale = 4

    opts1 = {
        'title': 'train_loss',
        'xlabel': 'epoch',
        'ylabel': 'loss',
        'width': 300,
        'height': 300,
    }

    opts2 = {
        'title': 'eval_psnr',
        'xlabel': 'epoch',
        'ylabel': 'psnr',
        'width': 300,
        'height': 300,
    }


opt = DefaultConfig()



