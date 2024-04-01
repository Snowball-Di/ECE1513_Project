from __future__ import print_function

from torch.utils.data import DataLoader
from dataset.data import get_training_set, get_test_set
from solver import DBPNTrainer
import torch.nn.functional as F

from evaluate import calculate_ssim, psnr

class Config:
    def __init__(self):
        self.batchSize = 4
        self.testBatchSize = 1
        self.nEpochs = 10
        self.lr = 0.001
        self.seed = 123
        self.upscale_factor = 4


    def __str__(self):
        return '\n'.join(f'{k} = {v}' for k, v in self.__dict__.items())

config = Config()


def bicubic_method(train_data_loader, testing_data_loader, scale_factor):
    avg_train_psnr = 0.0
    avg_train_ssim = 0.0
    for batch_num, (data, target) in enumerate(train_data_loader):
        bicubic_result = F.interpolate(data, scale_factor=scale_factor, mode='bicubic', align_corners=False)
        avg_train_psnr += psnr(target.cpu().detach().numpy(), bicubic_result.cpu().detach().numpy())
        avg_train_ssim += calculate_ssim(target, bicubic_result)
    avg_train_psnr /= (batch_num+1)
    avg_train_ssim /= (batch_num+1)
    avg_test_psnr = 0.0
    avg_test_ssim = 0.0
    for batch_num, (data, target) in enumerate(testing_data_loader):
        bicubic_result = F.interpolate(data, scale_factor=scale_factor, mode='bicubic', align_corners=False)
        avg_test_psnr += psnr(target.cpu().detach().numpy(), bicubic_result.cpu().detach().numpy())
        avg_test_ssim += calculate_ssim(target, bicubic_result)
    avg_test_psnr /= (batch_num+1)
    avg_test_ssim /= (batch_num+1)
    print(f'Bicubic Method Baseline: {avg_train_psnr=}, {avg_test_psnr=}, {avg_train_ssim=}, {avg_test_ssim}')
    

def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = get_training_set(config.upscale_factor)
    test_set = get_test_set(config.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, batch_size=config.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=config.testBatchSize, shuffle=False)


    model = DBPNTrainer(config, training_data_loader, testing_data_loader)
    model.run()
    bicubic_method(training_data_loader, testing_data_loader, 4)


if __name__ == '__main__':
    main()