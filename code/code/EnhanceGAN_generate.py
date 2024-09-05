from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../data/', help='path to dataset')
parser.add_argument('--dataname', default='abalone_train.csv', help='file name of dataset')
parser.add_argument('--outf', default='../output/model/', help='output path of GAN')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed',default=123, type=int, help='manual seed')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
parser.add_argument('--clamp_lower', type=float, default=-0.07)
parser.add_argument('--clamp_upper', type=float, default=0.07)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--hid_num', type=int, default=128, help='node number of hidden layer')
parser.add_argument('--noise_size', type=int, default=100, help='node number of hidden layer')
parser.add_argument('--num_discriminators', type=int, default=5, help='node number of discriminator for ensemble learning')
parser.add_argument('--optimizer', default='RMSprop', help='node number of hidden layer')

opt = parser.parse_args()

class CSVDataset(Dataset):
    """
    Custom dataset class for handling CSV data.

    This class loads a CSV file as a dataframe, extracts inputs and outputs,
    and provides functionality to retrieve data samples and labels.

    Args:
        filename (str): The path to the CSV file.
        class_label (str): The class label to filter the dataset.

    Attributes:
        data (list): List containing the input data samples.
        label (list): List containing the output labels.
        X (numpy.ndarray): Array containing the input data samples.
        y (numpy.ndarray): Array containing the output labels.

    """
    # load the dataset
    def __init__(self, filename,class_label):
        # load the csv file as a dataframe
        df = pd.read_csv(filename, header=None)
        # store the inputs and outputs
        self.data = (df.values[:, :-1]).tolist()
        self.lable = (df.values[:, -1]).tolist()

        self.X= []
        self.y=[]

        for i in range(0,len(self.lable)):
            if self.lable[i]==class_label:
                self.X.append(self.data[i])
                self.y.append(self.lable[i])

        self.X=np.array(self.X)
        self.y=np.array(self.y)

        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_dim(self):
        return len(self.X[0])

def read(filename):
    """
    Reads a CSV file and creates a list of CSVDataset objects, one for each unique class label.

    This function reads the CSV file, extracts the unique class labels, and creates
    a CSVDataset object for each class label. Each CSVDataset contains data filtered by the class label.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - list of CSVDataset objects, one for each unique class label.
            - list of unique class labels.
    """
    # Load the CSV file as a dataframe
    df = pd.read_csv(filename, header=None)

    # Extract and sort the unique class labels
    label_list = (df.values[:, -1]).tolist()
    label_list = sorted(list(set(label_list)), reverse=False)

    # Create a list of CSVDataset objects, one for each unique class label
    dataset_list = []
    for label in label_list:
        dataset_tmp = CSVDataset(filename, label)
        dataset_list.append(dataset_tmp)

    return dataset_list, label_list

def save_model(outf, data_i, netD_lis, netG):
    """
    Save the trained GAN models to the specified output directory.

    Args:
    - outf (str): The output directory where the models will be saved.
    - data_i (int): The index representing the class for which the models are trained.
    - netD_lis (list): List of discriminator networks.
    - netG (Generator): Generator network.

    Saves:
    - netG state_dict: Saved as 'netG_class_i.pth' where i is the class index.
    - netD state_dicts: Saved as 'netD_num_j_class_i.pth' where j is the discriminator index and i is the class index.
    """
    # Save the Generator network state_dict
    torch.save(netG.state_dict(), '%s/netG_class_%d.pth' % (outf, data_i + 1))

    # Save all discriminator networks
    for Dsave_i in range(opt.num_discriminators):
        torch.save(netD_lis[Dsave_i].state_dict(), '%s/netD_num_%d_class_%d.pth' % (outf, Dsave_i, data_i + 1))


def train_GAN(data_lis,data_i):
    """
       Trains a Generative Adversarial Network (GAN) using the provided data.

       This function trains a GAN model using the given list of datasets for a specific class label.

       Args:
           data_lis (list): A list of CSVDataset objects, each containing data samples for a specific class label.
           data_i (int): Index indicating the class label for which the GAN is trained.

       """
    # Some parameter settings
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    outf = opt.outf
    try:
        os.makedirs(outf)
    except OSError:
        pass
    #data in a single class
    dataset = data_lis[data_i]
    assert dataset

    batchSize = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,shuffle=True, num_workers=0)

    use_mps = opt.mps and torch.backends.mps.is_available()
    if opt.cuda:
        device = torch.device("cuda:0")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    #define the structure of generative adversarial network
    ngpu = int(opt.ngpu)
    data_dim=dataset.get_dim()
    nz=opt.noise_size
    ngf=opt.hid_num
    ndf=opt.hid_num
    nx=int(data_dim)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    #Generator network class
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.Linear(nz, ngf),
                nn.ReLU(True),
                nn.Linear(ngf, nx),

            )

        def forward(self, input):
            input = input.view(input.size(0), input.size(1))
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output

    # Discriminator network class
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                nn.Linear(nx, ndf),
                nn.ReLU(True),
                nn.Linear(ndf, 1),
                # nn.Sigmoid()
            )

        def forward(self, input):
            input = input.view(input.size(0),input.size(1))
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            output = output.mean(0)
            return output.view(1)

    #generator
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator with ensemble learning
    netD_lis = []
    for _ in range(opt.num_discriminators):
        netD_tmp = Discriminator(ngpu).to(device)
        netD_tmp.apply(weights_init)
        netD_lis.append(netD_tmp)

    # setup optimizer
    if opt.optimizer=='RMSprop':
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)
        optimizerD_lis=[]
        for D_i in range(opt.num_discriminators):
            optimizerD_tmp = optim.RMSprop(netD_lis[D_i].parameters(), lr=opt.lr)
            optimizerD_lis.append(optimizerD_tmp)
    elif opt.optimizer=='Adam':
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)
        optimizerD_lis = []
        for D_i in range(opt.num_discriminators):
            optimizerD_tmp = optim.Adam(netD_lis[D_i].parameters(), lr=opt.lr)
            optimizerD_lis.append(optimizerD_tmp)

    # define input
    input = torch.FloatTensor(batchSize, data_dim).to(device)
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    mone=mone.to(device)
    if opt.dry_run:
        opt.niter = 1

    # training the model
    gen_iterations = 0
    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for netD_tmp in netD_lis:
                for p in netD_tmp.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for netD_tmp in netD_lis:
                    for p in netD_tmp.parameters():
                        p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                ii=0
                errD_lis = []
                D_x_sum = []
                D_G_z1_sum = []
                for netD in netD_lis:
                    noise = torch.FloatTensor(batchSize, nz, 1, 1).to(device)
                    real_cpu, _ = data
                    netD.zero_grad()

                    if opt.cuda:
                        real_cpu = real_cpu.to(device)
                    input.resize_as_(real_cpu).copy_(real_cpu)
                    inputv = Variable(input)

                    errD_real = netD(inputv)
                    errD_real.backward(one)

                    # train with fake
                    noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
                    with torch.no_grad():
                        noisev = Variable(noise)  # totally freeze netG
                    fake = Variable(netG(noisev).data)
                    inputv = fake
                    errD_fake = netD(inputv)
                    errD_fake.backward(mone)
                    errD = errD_real - errD_fake
                    optimizerD_lis[ii].step()

                    errD_lis.append(errD)
                    ii += 1

                    output = netD(real_cpu)
                    D_x = output.mean().item()
                    D_x_sum.append(D_x)

                    output = netD(fake.detach())
                    D_G_z1 = output.mean().item()
                    D_G_z1_sum.append(D_G_z1)
                # softmax combination
                errD_avg = torch.tensor(errD_lis)
                D_x_sum = torch.tensor(D_x_sum)
                D_G_z1_sum = torch.tensor(D_G_z1_sum)

                softmax_2 = nn.Softmax(dim=-1)

                errD_weight = softmax_2(errD_avg)
                errD_tmp = torch.mul(errD_avg, errD_weight)
                errD_end = errD_tmp.mean(dim=-1)

                D_x_sum_weight = softmax_2(D_x_sum)
                D_x_sum_tmp = torch.mul(D_x_sum, D_x_sum_weight)
                D_x_avg = D_x_sum_tmp.mean(dim=-1)

                D_G_z1_sum_weight = softmax_2(D_G_z1_sum)
                D_G_z1_sum_tmp = torch.mul(D_G_z1_sum, D_G_z1_sum_weight)
                D_G_z1_avg = D_G_z1_sum_tmp.mean(dim=-1)


            ############################
            # (2) Update G network
            ###########################
            for netD_tmp in netD_lis:
                for p in netD_tmp.parameters():
                    p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            #softmax combination
            softmax_1 = nn.Softmax(dim=0)
            output_lis = []
            for netD_tmp in netD_lis:
                output_lis.append(netD_tmp(fake))
            output_lis = torch.stack(output_lis, 0)

            output_weight = softmax_1(output_lis)
            output_tmp = torch.mul(output_lis, output_weight)
            output = output_tmp.mean(dim=0)

            errG = output
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1
            D_G_z2 = output.mean().item()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                    errD_end.item(), errG.item(), D_x_avg, D_G_z1_avg, D_G_z2))
            log_txt=str('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                    errD_end.item(), errG.item(), D_x_avg, D_G_z1_avg, D_G_z2))
            f = open(outf+'log.txt', 'a', encoding='UTF-8')
            f.write(str(log_txt))
            f.write(str('\n'))

            if opt.dry_run:
                break
    #save the model
    save_model(outf,data_i,netD_lis,netG)



if __name__ == "__main__":
    """
        Main execution script for training GAN models for each class in the dataset.
        The GAN training for each class is performed sequentially, and the trained models
        are saved to the specified output directory.
    """
    # loading data
    filename=opt.dataroot+opt.dataname
    data_lis, lable_lis = read(filename)

    #training GAN for every class
    for data_i in range(0, len(data_lis)):
        train_GAN(data_lis, data_i)



