"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

from patchcore.tool import DatasetLoader
from patchcore.tool import UNet, Autoencoder,VAE

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def coordinate_dimension(features, target_channel):
    modified_features = features.copy()
    for i in range(len(modified_features)):
        if modified_features[i].shape[2] == target_channel:
            _features = modified_features[i]
            pass
        
        else:
            _features = modified_features[i]
            patch_size = modified_features[i].shape[2]
            _features = _features.reshape(len(modified_features[i]),1,-1)
            pooled_tensor = F.adaptive_avg_pool1d(_features, target_channel * (patch_size **2))
            _features = pooled_tensor.view(len(modified_features[i]),-1, patch_size, patch_size)
        
        modified_features[i] = _features.reshape(len(_features), -1)
        # modified_features[i] = _features.reshape(len(_features), -1, patch_size * patch_size)
        
    return modified_features

def list_concat(features, dim):
    if len(features) > 1:
        for i in range(len(features)):
            if i == 0:
                total_features = features[i]
            else:
                total_features = torch.cat((total_features, features[i]), dim = dim)
        return total_features
    
    elif len(features) <= 1:
        return torch.tensor(features[0])

def custom_preprocessing(features, target_dimension, dim = "patch"):
    if dim == "patch":
        _features = []
        for i in range(len(features)):
            pooled_tensor = features[i].reshape(features[i].shape[0], 1 , -1)
            pooled_tensor = F.adaptive_avg_pool1d(pooled_tensor, target_dimension).squeeze(1)
            
            _features.append(pooled_tensor)
            
        return _features
    
def make_sampling_and_dataset(features,
                              sampler,
                              device,
                              ):
    sampling_features, sample_indice, cluster_indice = sampler.run(features)
    
    sample_indice_tensor = torch.tensor(sample_indice).to(device)
    cluster_indice_tensor = [torch.tensor(cluster).to(device) for cluster in cluster_indice]
    

    for i in range(len(sample_indice)):
        input_tensor = features[sample_indice_tensor[i]]
        output_tensor = features[cluster_indice_tensor[i]]
        
        input_tensors = input_tensor.repeat(len(cluster_indice_tensor[i]), 1)
        if i == 0:
            total_input = input_tensors
            total_output = output_tensor
        else:
            total_input = torch.cat((total_input, input_tensors), dim = 0)
            total_output = torch.cat((total_output, output_tensor), dim = 0)
    
    return sampling_features, total_input, total_output

def make_test_dataset(model, features):
    # model.eval()
    # features = coordinate_dimension(features, 1024)
    # features = list_concat(features, 0)
    _features, _ , _= model(features)
    # layer1 = _features[int(_features.shape[0]//2):, :]
    # layer2 = _features[:int(_features.shape[0]//2), :]
    
    # layer = [layer1, layer2]
    # _features = custom_preprocessing(layer, 1024, "patch")
    
    return _features

def train(model, 
          device,
          input_data,
          output_data,
          epochs: int = 10,
          batch_size: int = 128,
          learning_rate: float = 1e-5,
          save_checkpoint: bool = False,
          amp: bool = False,
          weight_decay: float = 1e-8,
          momentum: float = 0.999,
          gradient_clipping: float = 1.0,
          lr_scheduler:str = "plateau"):
    
    dataset = DatasetLoader(input_data, output_data)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               pin_memory = True)
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
    
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target_data in train_loader:
                inputs = inputs.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    # outputs = model(inputs)
                    # loss = criterion(outputs, target_data)
                    
                    outputs, mu, logvar = model(inputs)
                    loss = model.vae_loss(outputs, inputs, mu, logvar, criterion)
                    loss.requires_grad_(True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                running_loss += loss.item() * inputs.size(0)  # 누적 손실

            scheduler.step(running_loss / len(dataset))  # 스케줄러에 평가 지표 전달
            print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(dataset)}")
            
            if save_checkpoint:
                # Checkpoint 저장 로직 추가
                pass
    elif lr_scheduler == "lambda":
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda=learning_rate_scheduler)
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target_data in train_loader:
                inputs = inputs.to(device)
                target_data = target_data.to(device)    
                optimizer.zero_grad()

                outputs, mu, logvar = model(inputs)
                loss = model.vae_loss(outputs, inputs, mu, logvar, criterion)
                loss.requires_grad_(True)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(input_data)}")   
                

def learning_rate_scheduler(epoch):
        
    return 0.1 ** (epoch//10)


class VAE(nn.Module):
    def __init__(self, shape):
        super().__init__()
        
        patch_size = int(shape[2])
        dimension = int(shape[1])
        
        self.encoder = nn.Sequential(
            nn.Linear(dimension, dimension//2),
            # nn.BatchNorm1d(dimension//2),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//2, dimension//(4)),
            # nn.BatchNorm1d(dimension//(4)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(4), dimension//(8)),
            # nn.BatchNorm1d(dimension//(8)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(8), dimension//(16)),
            # nn.BatchNorm1d(dimension//(16)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(16), dimension//(16)),
            # nn.BatchNorm1d(dimension//(16)),
            nn.ReLU(inplace=True)
        )
        
        self.fc_mu = nn.Linear(dimension//(16), dimension//(32))
        self.fc_logbar = nn.Linear(dimension//(16), dimension//(32))
        
        self.decoder = nn.Sequential(
            nn.Linear(dimension//(32), dimension//(16)),
            # nn.BatchNorm1d(dimension//(16)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(16), dimension//(8)),
            # nn.BatchNorm1d(dimension//(8)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(8), dimension//(4)),
            # nn.BatchNorm1d(dimension//(4)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(4), dimension//(2)),
            # nn.BatchNorm1d(dimension//(2)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(2), dimension),
            # nn.BatchNorm1d(dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, dimension)
        )
        self.to(device)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
        
    def forward(self, x):
        latent = self.encoder(x)
        mu, logvar = self.fc_mu(latent), self.fc_logbar(latent)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar
    
    def vae_loss(self, x_recon, x, mu, logvar, criterion):
        reconstruction_loss = criterion(x_recon,x)
        kl_divergence = -0.5 *torch.sum(1+logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence

class Autoencoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        
        patch_size = int(shape[2])
        dimension = int(shape[1]) * (patch_size**2)
        
        self.encoder = nn.Sequential(
            nn.Linear(dimension, dimension//(patch_size)),
            # nn.BatchNorm1d(dimension//(patch_size)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(patch_size), dimension//(patch_size**2)),
            # nn.BatchNorm1d(dimension//(patch_size**2)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//(patch_size**2), dimension//((patch_size**2)*4)),
            # nn.BatchNorm1d(dimension//((patch_size**2)*4)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//((patch_size**2)*4), dimension//((patch_size**2)*16)),
            nn.BatchNorm1d(dimension//((patch_size**2)*16)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//((patch_size**2)*16), dimension//((patch_size**2)*16)),
            nn.BatchNorm1d(dimension//((patch_size**2)*16)),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(dimension//((patch_size**2)*16), dimension//((patch_size**2)*4)),
            nn.BatchNorm1d(dimension//((patch_size**2)*4)),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//((patch_size**2)*4), dimension//((patch_size**2))),
            nn.BatchNorm1d(dimension//((patch_size**2))),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//((patch_size**2)), dimension//((patch_size))),
            # nn.BatchNorm1d(dimension//((patch_size))),
            nn.ReLU(inplace=True),
            nn.Linear(dimension//((patch_size)), dimension),
            # nn.BatchNorm1d(dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, dimension)
        )
        self.to(device)
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 512))
        self.down1 = (Down(512, 256))
        self.down2 = (Down(256, 128))
        self.down3 = (Down(128, 64))
        factor = 2 if bilinear else 1
        self.down4 = (Down(64, 32 // factor))
        self.up1 = (Up(32, 64 // factor, bilinear))
        self.up2 = (Up(64, 128 // factor, bilinear))
        self.up3 = (Up(128, 256 // factor, bilinear))
        self.up4 = (Up(256, 512, bilinear))
        self.outc = (OutConv(512, n_channels))

    def forward(self, x):
        # x.shape = 256, 1024, 3, 3
        # x1.shape = 256, 512, 3, 3
        # x2.shape = 256, 256, 3, 3
        # x3.shape = 256, 128, 3, 3
        # x4.shape = 256, 64, 3, 3
        # x5.shape = 256, 32, 3, 3
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        image = self.outc(x)
        return image


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            ### modified code, if we want to run maxpool, should extend patch_size
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels * 2, kernel_size=1, stride=1)
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
            self.conv = DoubleConv(in_channels * 4, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
   
class VAE_Linear(nn.Module):
    def __init__(self, features, layer):
        super().__init__()
        
        dimension = features.shape[1]
        self.latent_dim = dimension // (2)
        if layer == 'linear':
            # Encoder layers
            self.encoder = nn.Sequential(
                nn.Linear(dimension, dimension // 2),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )
            # Latent layers
            self.fc_mu = nn.Linear(dimension // 2, self.latent_dim)
            self.fc_logvar = nn.Linear(dimension // 2, self.latent_dim)
            
            # Decoder layers
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, dimension // 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(dimension // 2, dimension),    
            )
        elif layer == 'conv':
            pass
            
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.learning_rate_scheduler)
        
        self.to(device)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def train(self, input_data, epochs, denoising=False):
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target_data in input_data:
                inputs = inputs.to(device)
                
                if denoising:
                    inputs = inputs + torch.normal(0, 0.1, size=inputs.size()).to(device)
                
                target_data = target_data.to(device)    
                self.optimizer.zero_grad()
                outputs, mu, logvar = self.forward(inputs)
                
                loss = self.vae_loss(outputs, target_data, mu, logvar)
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(input_data)}")   
                
    def learning_rate_scheduler(self, epoch):
        return 0.1 ** (epoch // 10)
    
    def vae_loss(self, x_recon, x, mu, logvar):
        reconstruction_loss = self.criterion(x_recon, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence



class V_Autoencoder(nn.Module):
    def __init__(self, features):
        super(V_Autoencoder, self).__init__()
        input_dim = features.shape[1]
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
        )
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.decoder = nn.Sequential(
        )
        
        self.to(device=device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        
        return mu + eps + std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def vae_loss(self, x_recon, x, mu, logvar,criterion):
        reconstruction_loss = criterion(x_recon, x)
        kl_divergence = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence
    
    def train_(self, 
              input_data, 
              output_data, 
              epochs,
              batch_size,
              learning_rate,
              ):
        
        data = DatasetLoader(input_data, output_data)
        train_loader = torch.utils.data.DataLoader(data,
                                    batch_size,
                                    shuffle=True,
                                    # pin_memory=True,
                                    )
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.learning_rate_scheduler)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target_data in train_loader:
                inputs = inputs.to(device)
                target_data = target_data.to(device)    
                optimizer.zero_grad()
                outputs, mu, logvar = self.forward(inputs)
                
                loss = self.vae_loss(outputs, target_data, mu, logvar, criterion)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(input_data)}")   
            
    def learning_rate_scheduler(self, epoch):
        return 0.1 ** (epoch // 10)
    

class kl_Autoencoder(nn.Module):
    def __init__(self, features):
        super(Autoencoder, self).__init__()
        input_dim = features.shape[1]
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
        )
        
        self.decoder = nn.Sequential(
        )
        
        self.to(device=device)
        
    def forward(self, x):
        h = self.encoder(x)
        recon_x = self.decoder(h)
        return recon_x
    
    def autoencoder_loss(self, recon_x, x, criterion):
        reconstruction_loss = criterion(recon_x, x)
        
        train_dist = self.estimate_distribution(x)
        recon_dist = self.estimate_distribution(recon_x)
        kl_loss = self.cal_kl_diverence(train_dist, recon_dist)
        
        return reconstruction_loss + kl_loss
    
    def estimate_distribution(self, data):
        data = data.unsqueeze(1)
        kernel_value = torch.kde.gaussian_kernel(data, data, kernel_width = 0.5)
        kde = torch.mean(kernel_value)
        
        distribution = kde/torch.sum(kde)
        
        return distribution
    
    def cal_kl_diverence(self, train, recon):
        kl_divergence = torch.sum(train * torch.log(train/recon))
        
        return kl_divergence
    
    def train_(self, 
              input_data, 
              output_data, 
              epochs,
              batch_size,
              learning_rate,
              ):
        
        data = DatasetLoader(input_data, output_data)
        train_loader = torch.utils.data.DataLoader(data,
                                    batch_size,
                                    shuffle=True,
                                    # pin_memory=True,
                                    )
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.learning_rate_scheduler)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target_data in train_loader:
                inputs = inputs.to(device)
                target_data = target_data.to(device)    
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                
                loss = self.autoencoder_loss(outputs, target_data, criterion)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(input_data)}")   
            
    def learning_rate_scheduler(self, epoch):
        return 0.1 ** (epoch // 10)

import torch
import numpy as np

from math import pi
from scipy.special import logsumexp
from patchcore.utils import calculate_matmul, calculate_matmul_n_times


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self._init_params()


    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
                    requires_grad=False
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)
        self.params_fitted = False


    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x


    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic


    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))


    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)


    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in np.arange(self.n_components)[counts > 0]: 
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y


    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score


    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            precision = torch.inverse(var)
            d = x.shape[-1]

            log_2pi = d * np.log(2. * pi)

            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * prec, dim=2, keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p - log_det)


    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)
        
        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0,k]))).sum()

        return log_det.unsqueeze(-1)


    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                            keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps

        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var


    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)


    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var


    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi


    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)
        
        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0)*(x_max - x_min) + x_min)
    