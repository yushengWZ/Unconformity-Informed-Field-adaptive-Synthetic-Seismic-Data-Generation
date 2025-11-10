import os
import argparse
import itertools
from datetime import datetime

import torch
from torch import nn

import dataset
import dataset_cyclegan
import imagepool
import rand_seed
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable

from lossf import loss
from model import Generator, Discriminator
import pytorch_ssim
import loss1

# rand_seed.rand_seed(100)

device = torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='train batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=5 * 1e-5, help='learning rate for generator')
parser.add_argument('--lrD', type=float, default=1 * 1e-4, help='learning rate for discriminator')
parser.add_argument('--weight_G_A', type=float, default=1, help='weight for G_A loss')
parser.add_argument('--weight_G_B', type=float, default=1, help='weight for G_B loss')
parser.add_argument('--weight_cycle_A', type=float, default=2, help='weight for cycle loss')
parser.add_argument('--weight_cycle_B', type=float, default=2, help='weight for cycle loss')
params = parser.parse_args()
print(params)

current_time = datetime.now().strftime('%b%d_%H%M%S')
images_dir = r'imgs'
model_dir = r'models'
logpathnew = os.path.join(model_dir, current_time + '_Finetune_0512_random_combine_nofilt_freeze_nolayer_dicessim_mse')
if not os.path.exists(images_dir):
    os.mkdir(images_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(logpathnew):
    os.mkdir(logpathnew)

# LOAD DATA

data_set = dataset_cyclegan.his_dataset_SYN2KERRYori()


train_loader = Data.DataLoader(dataset=data_set, batch_size=params.batch_size, shuffle=True)

# Models
G_A = Generator(in_channels=1, n_classes=1, depth=4, wf=5, padding=True, batch_norm=True, up_mode='upconv')
G_B = Generator(in_channels=1, n_classes=1, depth=4, wf=4, padding=True, batch_norm=True, up_mode='upconv')
D_A = Discriminator(input_dims=1, hidden_dims=32, output_dims=1)
D_B = Discriminator(input_dims=1, hidden_dims=32, output_dims=1)

G_A.to(device)
G_B.to(device)
D_A.to(device)
D_B.to(device)

# for name, param in G_A.named_parameters():
#     print(name)
#
# print('2\n')
#
# for name, param in G_B.named_parameters():
#     print(name)

freeze_layers = {
    'down_path.0.block.0.weight',
    'down_path.0.block.0.bias',
    'down_path.0.block.2.weight',
    'down_path.0.block.2.bias',
    'down_path.0.block.3.weight',
    'down_path.0.block.3.bias',
    'down_path.0.block.5.weight',
    'down_path.0.block.5.bias',
    'down_path.1.block.0.weight',
    'down_path.1.block.0.bias',
    'down_path.1.block.2.weight',
    'down_path.1.block.2.bias',
    'down_path.1.block.3.weight',
    'down_path.1.block.3.bias',
    'down_path.1.block.5.weight',
    'down_path.1.block.5.bias',
    'down_path.2.block.0.weight',
    'down_path.2.block.0.bias',
    'down_path.2.block.2.weight',
    'down_path.2.block.2.bias',
    'down_path.2.block.3.weight',
    'down_path.2.block.3.bias',
    'down_path.2.block.5.weight',
    'down_path.2.block.5.bias',
    'down_path.3.block.0.weight',
    'down_path.3.block.0.bias',
    'down_path.3.block.2.weight',
    'down_path.3.block.2.bias',
    'down_path.3.block.3.weight',
    'down_path.3.block.3.bias',
    'down_path.3.block.5.weight',
    'down_path.3.block.5.bias',
    # 'up_path.0.up.weight',
    # 'up_path.0.up.bias',
    # 'up_path.0.conv_block.block.0.weight',
    # 'up_path.0.conv_block.block.0.bias',
    # 'up_path.0.conv_block.block.2.weight',
    # 'up_path.0.conv_block.block.2.bias',
    # 'up_path.0.conv_block.block.3.weight',
    # 'up_path.0.conv_block.block.3.bias',
    # 'up_path.0.conv_block.block.5.weight',
    # 'up_path.0.conv_block.block.5.bias',
    # 'up_path.1.up.weight',
    # 'up_path.1.up.bias',
    # 'up_path.1.conv_block.block.0.weight',
    # 'up_path.1.conv_block.block.0.bias',
    # 'up_path.1.conv_block.block.2.weight',
    # 'up_path.1.conv_block.block.2.bias',
    # 'up_path.1.conv_block.block.3.weight',
    # 'up_path.1.conv_block.block.3.bias',
    # 'up_path.1.conv_block.block.5.weight',
    # 'up_path.1.conv_block.block.5.bias',
    # 'up_path.2.up.weight',
    # 'up_path.2.up.bias',
    # 'up_path.2.conv_block.block.0.weight',
    # 'up_path.2.conv_block.block.0.bias',
    # 'up_path.2.conv_block.block.2.weight',
    # 'up_path.2.conv_block.block.2.bias',
    # 'up_path.2.conv_block.block.3.weight',
    # 'up_path.2.conv_block.block.3.bias',
    # 'up_path.2.conv_block.block.5.weight',
    # 'up_path.2.conv_block.block.5.bias',
    # 'last.weight',
    # 'last.bias',
}

freeze_layers_B = {
    'down_path.0.block.0.weight',
    'down_path.0.block.0.bias',
    'down_path.0.block.2.weight',
    'down_path.0.block.2.bias',
    'down_path.0.block.3.weight',
    'down_path.0.block.3.bias',
    'down_path.0.block.5.weight',
    'down_path.0.block.5.bias',
    'down_path.1.block.0.weight',
    'down_path.1.block.0.bias',
    'down_path.1.block.2.weight',
    'down_path.1.block.2.bias',
    'down_path.1.block.3.weight',
    'down_path.1.block.3.bias',
    'down_path.1.block.5.weight',
    'down_path.1.block.5.bias',
    'down_path.2.block.0.weight',
    'down_path.2.block.0.bias',
    'down_path.2.block.2.weight',
    'down_path.2.block.2.bias',
    'down_path.2.block.3.weight',
    'down_path.2.block.3.bias',
    'down_path.2.block.5.weight',
    'down_path.2.block.5.bias',
    'down_path.3.block.0.weight',
    'down_path.3.block.0.bias',
    'down_path.3.block.2.weight',
    'down_path.3.block.2.bias',
    'down_path.3.block.3.weight',
    'down_path.3.block.3.bias',
    'down_path.3.block.5.weight',
    'down_path.3.block.5.bias',
    # 'up_path.0.up.weight',
    # 'up_path.0.up.bias',
    # 'up_path.0.conv_block.block.0.weight',
    # 'up_path.0.conv_block.block.0.bias',
    # 'up_path.0.conv_block.block.2.weight',
    # 'up_path.0.conv_block.block.2.bias',
    # 'up_path.0.conv_block.block.3.weight',
    # 'up_path.0.conv_block.block.3.bias',
    # 'up_path.0.conv_block.block.5.weight',
    # 'up_path.0.conv_block.block.5.bias',
    # 'up_path.1.up.weight',
    # 'up_path.1.up.bias',
    # 'up_path.1.conv_block.block.0.weight',
    # 'up_path.1.conv_block.block.0.bias',
    # 'up_path.1.conv_block.block.2.weight',
    # 'up_path.1.conv_block.block.2.bias',
    # 'up_path.1.conv_block.block.3.weight',
    # 'up_path.1.conv_block.block.3.bias',
    # 'up_path.1.conv_block.block.5.weight',
    # 'up_path.1.conv_block.block.5.bias',
    # 'up_path.2.up.weight',
    # 'up_path.2.up.bias',
    # 'up_path.2.conv_block.block.0.weight',
    # 'up_path.2.conv_block.block.0.bias',
    # 'up_path.2.conv_block.block.2.weight',
    # 'up_path.2.conv_block.block.2.bias',
    # 'up_path.2.conv_block.block.3.weight',
    # 'up_path.2.conv_block.block.3.bias',
    # 'up_path.2.conv_block.block.5.weight',
    # 'up_path.2.conv_block.block.5.bias',
    # 'last.weight',
    # 'last.bias',
}

# todo:kerry_mse
G_A.load_state_dict(torch.load(
    r'models\generator_A_param_Epoch200_step120.pkl'))
G_B.load_state_dict(torch.load(
    r'models\generator_B_param_Epoch200_step120.pkl'))



for name, param in G_A.named_parameters():
    # print(name)
    if name in freeze_layers:
        param.requires_grad = False
        print(f'{name}:param.requires_grad:{param.requires_grad}')

for name, param in G_B.named_parameters():
    # print(name)
    if name in freeze_layers_B:
        param.requires_grad = False
        print(f'{name}:param.requires_grad:{param.requires_grad}')


class l1ssim(nn.Module):

    def __init__(self):
        super(l1ssim, self).__init__()
        self.dice = torch.nn.L1Loss()
        # self.ssim = 1 - pytorch_ssim.ssim(recon_B, real_B)
        self.name = 'l1ssim'
        self.a = 0.5
        self.b = 0.5

    def forward(self, y_true, y_pred):
        d = self.dice(y_true, y_pred)
        c = 1 - pytorch_ssim.ssim(y_true, y_pred)
        loss = self.a * d + self.b * c
        return loss

    def getLossName(self):
        return self.name


# Loss function
MSE_loss = torch.nn.MSELoss().to(device)
L1_loss = torch.nn.L1Loss().to(device)
dice_loss = loss1.diceloss().to(device)
dice_ssim_loss = loss.dicessim(0.5, 0.5).to(device)

# optimizers
G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=params.lrG)
D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params.lrD)
D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params.lrD)

# Generated image pool
num_pool = 100
fake_A_pool = imagepool.ImagePool(num_pool)
fake_B_pool = imagepool.ImagePool(num_pool)


for epoch in range(params.num_epochs):
    for step, (real_A, real_B) in enumerate(train_loader):

        # ************* Train A -> B ************ #
        real_A = Variable(real_A.to(device))
        fake_B = G_A(real_A)
        D_B_fake_decision = D_B(fake_B)
        G_A_loss = MSE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).to(device)))

        recon_A = G_B(fake_B)
        # cycle_A_loss = dice_loss(recon_A, real_A)
        # cycle_A_loss = L1_loss(recon_A, real_A)
        cycle_A_loss = dice_ssim_loss(recon_A, real_A)
        # cycle_A_loss = 1 - pytorch_ssim.ssim(recon_A, real_A)

        # ************* Train B -> A ************ #
        real_B = Variable(real_B.to(device))
        fake_A = G_B(real_B)
        D_A_fake_decision = D_A(fake_A)
        G_B_loss = MSE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).to(device)))

        recon_B = G_A(fake_A)
        # cycle_B_loss = L1_loss(recon_B, real_B)
        cycle_B_loss = MSE_loss(recon_B, real_B)
        # cycle_B_loss = l1ssim()(recon_B, real_B)
        # cycle_B_loss = 1 - pytorch_ssim.ssim(recon_B, real_B)

        G_loss = G_A_loss * params.weight_G_A + cycle_A_loss * params.weight_cycle_A + \
                 G_B_loss * params.weight_G_B + cycle_B_loss * params.weight_cycle_B

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()


        # ************* Train discriminator D_A ************ #
        D_A_real_decision = D_A(real_A)
        D_A_real_loss = MSE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).to(device)))
        fake_A = fake_A_pool.query(fake_A)
        D_A_fake_decision = D_A(fake_A)
        D_A_fake_loss = MSE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).to(device)))

        # Back propagation
        D_A_loss = (D_A_real_loss + D_A_fake_loss)
        D_A_optimizer.zero_grad()
        D_A_loss.backward()
        D_A_optimizer.step()

        # ************* Train discriminator D_B ************ #
        D_B_real_decision = D_B(real_B)
        D_B_real_loss = MSE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).to(device)))
        fake_B = fake_B_pool.query(fake_B)
        D_B_fake_decision = D_B(fake_B)
        D_B_fake_loss = MSE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).to(device)))

        # Back propagation
        D_B_loss = (D_B_real_loss + D_B_fake_loss)
        D_B_optimizer.zero_grad()
        D_B_loss.backward()
        D_B_optimizer.step()


        if (step + 1) % 20 == 0:
            print('Epoch [%d/%d], Step [%d/%d], '
                  'G_A_loss: %.4f, cycle_A_loss: %.4f, G_B_loss: %.4f, cycle_B_loss: %.4f, '
                  'G_loss: %.4f, D_A_loss: %.4f, D_B_loss: %.4f' %
                  (epoch + 1, params.num_epochs, step + 1, len(train_loader),
                   G_A_loss.item(), cycle_A_loss.item(), G_B_loss.item(), cycle_B_loss.item(),
                   G_loss.item(), D_A_loss.item(), D_B_loss.item()))

        if epoch == params.num_epochs - 1:
            # if step == len(train_loader) - 1:
            if (step + 1) % 20 == 0:
                print('Epoch [%d/%d], Step [%d/%d], '
                      'G_A_loss: %.4f, cycle_A_loss: %.4f, G_B_loss: %.4f, cycle_B_loss: %.4f, '
                      'G_loss: %.4f, D_A_loss: %.4f, D_B_loss: %.4f' %
                      (epoch + 1, params.num_epochs, step + 1, len(train_loader),
                       G_A_loss.item(), cycle_A_loss.item(), G_B_loss.item(), cycle_B_loss.item(),
                       G_loss.item(), D_A_loss.item(), D_B_loss.item()))
                save_name = os.path.join(logpathnew, 'generator_A_param_' + 'Epoch' + str(epoch + 1) + '_step' + str(
                    step + 1) + '.pkl')
                torch.save(G_A.state_dict(),
                           save_name)
