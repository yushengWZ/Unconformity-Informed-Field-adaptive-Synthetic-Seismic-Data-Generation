import torch
import dataset_cyclegan
import numpy as np
import torch.utils.data as Data
from model import Generator
from datetime import datetime

current_time = datetime.now().strftime('%b%d_%H%M%S')
# load data


# todo:best until now
data_set = dataset_cyclegan.his_dataset_SYN2KERRYori()

train_loader = Data.DataLoader(dataset=data_set, batch_size=1, shuffle=False)

# CycleGAN model
G = Generator(in_channels=1, n_classes=1, depth=4, wf=5, padding=True, batch_norm=True, up_mode='upconv')

G.cuda()

# todo:finetune
# MSE
G.load_state_dict(torch.load(
    r'models\generator_A_param_Epoch200_step120.pkl'))

output_seis_images = np.zeros([4000, 1, 128, 128])
input_images = np.zeros([4000, 1, 128, 128])
target_images = np.zeros([4000, 1, 128, 128])

# training
for step, (input_labels, _,) in enumerate(train_loader):
    print('complete...%d/%d' % (int(step + 1), int(len(train_loader))))
    input_labels = input_labels.cuda()
    target_images1 = _.cpu().numpy()
    seis_images = G(input_labels)
    seis_images = seis_images.detach().cpu()
    seis_images = (seis_images - seis_images.mean()) / seis_images.std()
    output_seis_images[step, 0, :, :] = seis_images
    input_labels1 = input_labels.cpu().numpy()
    input_images[step, 0, :, :] = input_labels1
    target_images[step, 0, :, :] = target_images1

print('finish!')

unconf = np.load(r'unconf.npy').transpose([2, 0, 1])
fault = np.load(r'fault.npy').transpose([2, 0, 1])
fault[fault > 1] = 1

import matplotlib.pyplot as plt

a = 200

plt.figure()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度
plt.yticks([])  # 去刻度
plt.imshow(unconf[a, :, :], cmap='gray')
plt.savefig(rf'E:\imgs\{a}_1_1.png', dpi=600, bbox_inches='tight', pad_inches=-0.01)
plt.show()

plt.figure()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度
plt.yticks([])  # 去刻度
plt.imshow(target_images[a, 0, :, :], cmap='gray')
plt.savefig(rf'E:\imgs\{a}_2.png', dpi=600, bbox_inches='tight', pad_inches=-0.01)
plt.show()
#
plt.figure()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度
plt.yticks([])  # 去刻度
plt.imshow(input_images[a, 0, :, :], cmap='gray')
plt.savefig(rf'E:\imgs\{a}_3_1.png', dpi=600, bbox_inches='tight', pad_inches=-0.01)
plt.show()
#

#
plt.figure()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度
plt.yticks([])  # 去刻度
plt.imshow(fault[a, :, :], cmap='gray')
plt.savefig(rf'E:\imgs\{a}_4_1.png', dpi=600, bbox_inches='tight', pad_inches=-0.01)
plt.show()
#
#
plt.figure()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度
plt.yticks([])  # 去刻度
plt.imshow(output_seis_images[a, 0, ...], cmap='gray')
plt.savefig(rf'E:\imgs\{a}_5_1.png', dpi=600, bbox_inches='tight', pad_inches=-0.01)
plt.show()
