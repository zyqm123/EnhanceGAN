import argparse
import copy
import csv
import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

import torch
from MLP import Generator, Discriminator
from calibration import LRcalibrator
from en_repgan import repgan

#-------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../data/', help='path to dataset')
parser.add_argument('--dataname', default='abalone_train.csv', help='file name of dataset')
parser.add_argument('--load-g', default='netG', help='name for the generator file')
parser.add_argument('--load-d', default='netD', help='name for the discriminator file')
parser.add_argument('--load_model', default='../output/model/', help='path for the discriminator and generator file')
parser.add_argument('--data', default='../data/', help='path for dataset')
parser.add_argument('--outf', default='../output/generatedata/', help='path to save the results of MCMC sampling')
parser.add_argument('--hid_num', type=int, default=128, help='node number of hidden layer')
parser.add_argument('--noise_size', type=int, default=100, help='node number of hidden layer')
parser.add_argument('--num_discriminators', type=int, default=5, help='node number of discriminator for ensemble learning')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus')
parser.add_argument('--calibrate', action='store_true',
                    help='whether to calibrate the discriminator scores (if true, use LR, else, use id mapping)')
parser.add_argument('--clen', type=int, default=640, help='length of each Markov chain')
parser.add_argument('--tau', type=float, default=0.60, help='Langevin step size in L2MC')
parser.add_argument('--eta', type=float, default=0.40, help='scale of white noise (default to sqrt(tau))')
parser.add_argument('--manualSeed', default=10000, type=int, help='manual seed')
parser.add_argument('--generate_num', default=3000, type=int, help='manual seed')
parser.add_argument('--calibrate_use', default=True)
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


def get_parameter(filename):
    """
    Extracts parameters and their corresponding values from a text file.

    Args:
    - filename (str): The path to the text file containing parameter names and values.

    Returns:
    - params (dict): A dictionary containing parameter names as keys and their values as values.
    """
    # Dictionary to store parameters
    params = {}

    # Read the text file
    with open(filename, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split each line by whitespace
            tokens = line.split()

            # Ensure each line contains a parameter name and value
            if len(tokens) == 2:
                # Extract parameter name and value
                param_name = tokens[0]
                param_value = tokens[1]

                # Store in the dictionary
                params[param_name] = param_value
    return params

def save_file(sample_dataset):
    """
    Saves the generated sample dataset to a CSV file.

    Args:
    - fi (str): The filename or identifier for the sample dataset.
    - sample_dataset (list): The sample dataset to be saved.

    Saves:
    - CSV file: The sample dataset is saved to a CSV file with the provided filename.
    """
    # Specify the CSV file path
    csv_path = opt.outf + opt.dataname

    # Write the sample dataset to the CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as s1:
        writer1 = csv.writer(s1)
        writer1.writerows(sample_dataset)


def MCMC(data_lis, lable_lis):
    """
    MCMC sampling method to generate synthetic data.

    Args:
    - data_lis (list): List of datasets.
    - lable_lis (list): List of labels.

    """

    # Setting random seed if not provided
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)

    sample_dataset = []

    # Iterating over each dataset
    for data_i in range(0, len(data_lis)):
        dataset = data_lis[data_i]

        # Calculating sample numbers based on dataset index
        if data_i == 0:
            samp_num = int(0.5 * opt.generate_num)
        else:
            samp_num = opt.generate_num - int(0.5 * opt.generate_num)

        model_path = opt.load_model
        save_path = opt.outf

        try:
            os.makedirs(save_path)
        except OSError:
            pass

        # Loading parameters of GAN
        nx = dataset.get_dim()
        nz = opt.noise_size
        ngf = opt.hid_num
        ndf = opt.hid_num
        num_D = opt.num_discriminators
        batchsize = len(dataset)

        # Loading the generator
        load_g = model_path + opt.load_g
        load_d = model_path + opt.load_d
        device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

        netG = Generator(opt.ngpu, nz, ngf, nx).to(device)
        netG.load_state_dict(torch.load(load_g + '_class_' + str(data_i + 1) + '.pth', map_location=device))

        # Loading the discriminator
        netD_lis = []
        for D_i in range(num_D):
            netD_tmp = Discriminator(opt.ngpu, ndf, nx).to(device)
            netD_tmp.load_state_dict(torch.load(load_d + '_num_' + str(D_i) + '_class_' + str(data_i + 1) + '.pth', map_location=device))
            netD_lis.append(netD_tmp)

        print('Model loaded')
        torch.set_grad_enabled(False)

        # Calibrating discriminator
        if opt.calibrate_use:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0)
            calibrator = LRcalibrator(netG, netD_lis, data_loader, device, nz=nz)
        else:
            calibrator = torch.nn.Identity()

        print('Start sampling')

        samples = repgan(netG, netD_lis, calibrator, device, nz, samp_num, opt.clen, opt.tau, opt.eta)
        accepted_samples = samples

        samples_lis = []
        for acc_data in accepted_samples:
            for accpte_sam in acc_data:
                tmp = accpte_sam.tolist()
                tmp1 = copy.deepcopy(tmp)
                samples_lis.append(tmp1)
                tmp.append(lable_lis[data_i])
                sample_dataset.append(tmp)

    save_file(sample_dataset)


if __name__ == "__main__":
    """
        Main execution script for training GAN models using MCMC sampling.
    """
    # loading data
    filename = opt.dataroot + opt.dataname
    data_lis, lable_lis = read(filename)

    # MCMC sampling
    MCMC(data_lis,lable_lis)


