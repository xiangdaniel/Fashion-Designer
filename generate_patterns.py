import argparse
import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import models.fashion_dcgan as dcgan

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='samples', type=str, help='trained model folder')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='number of discrim filters in first conv layer')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--which_epoch', default='24', type=str, help='0,1,2,3,4...')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
opt = parser.parse_args()
print(opt)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)  # 3
n_extra_layers = int(opt.n_extra_layers)

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
ngpu = len(gpu_ids)

try:
    os.makedirs(os.path.join('./results_fashion', opt.name))
except OSError:
    pass


def truncation(fixed_noise):
    for i in range(len(fixed_noise)):
         v = fixed_noise[i]
         while abs(v)>2:
             v= torch.randn(1)
         fixed_noise[i] = v
    return fixed_noise


# generate images
def generate_img(model):
    for i in range(3):
        input_noise = torch.FloatTensor(opt.batchsize*100).normal_(0, 1)
        input_noise = truncation(input_noise)
        input_noise = input_noise.resize_(opt.batchsize, nz, 1, 1)
        input_noise = input_noise.cuda()
        input_noise = Variable(input_noise)
        outputs = model(input_noise)
        fake = outputs.data
        print(fake.shape)
        for j in range(opt.batchsize):
            im = fake[j, :, :, :]
            torchvision.utils.save_image(
                    im.view(1, im.size(0), im.size(1), im.size(2)),
                    os.path.join('./results_fashion', '%d_%d.png' % (i, j)),
                    nrow=1,
                    padding=0,
                    normalize=True)


# Load model
def load_network(network):
    save_path = os.path.join('./', opt.name, 'netG_epoch_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


if __name__ == '__main__':
    model_structure = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    model = load_network(model_structure)
    model = model.cuda()
    generate_img(model)
