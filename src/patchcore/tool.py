import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import random
import numpy as np
from scipy.ndimage import rotate

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        super(DatasetLoader, self).__init__()
        self.transform = transform
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
     
        if self.transform:
            data = self.transform(data)
            # Try : target = data
            target = self.transform(target)
        
        return data, target
    
class Autoencoder(nn.Module):
    def __init__(self, features, patch_size):
        super(Autoencoder, self).__init__()
        self.patch_size = patch_size
        input_dim = features.shape[1]
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU() 
        )
        
        self.decoder = nn.Sequential(
        )
        
        self.to(device=device)
        
    def forward(self, x):
        h = self.encoder(x)
        recon_x = self.decoder(h)
        return recon_x
    
    def autoencoder_loss(self, x_recon, x, criterion):
        reconstruction_loss = criterion(x_recon, x)
        return reconstruction_loss
    
    def train_(self, 
              input_data, 
              output_data, 
              epochs,
              batch_size,
              learning_rate,
              ):
        
        # random_rotate_patches = RandomRotatePatches(self.patch_size)(input_data)

        # Rotate transform
        # data = DatasetLoader(input_data, output_data, transform=random_rotate_patches)
        data = DatasetLoader(input_data, output_data, transform=None)
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
            # print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(input_data)}")   
            
    def learning_rate_scheduler(self, epoch):
        return 0.1 ** (epoch // 10)

def make_sampling_ratio(dataset, threshold, dimension):
    dataset = torch.tensor(dataset, device = device)
    if dimension == dataset.shape[1]:
        dataset_mapped = dataset.clone()
        pass
    
    else:        
        mapper = torch.nn.Linear(
            dataset.shape[1], dimension, bias=False
        )
        _ = mapper.to(device)
        dataset_mapped = mapper(dataset)

    def cosine_similarity(vec1, vec2):
        similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(1), dim=2)
        return similarity

    selected_datas = []
    length = len(dataset)
    i = 0
    while len(dataset_mapped) > 0:
        # i = i+1
        # print(i)
        selected_data_index = random.randint(0, len(dataset_mapped) - 1)
        selected_data = dataset_mapped[selected_data_index]
        
        # if i != 1:
        #     repeated_length = length - len(dataset_mapped)
        #     repeated_tensor = selected_data.repeat(repeated_length, 1)
        #     dataset_mapped = torch.cat((dataset_mapped, repeated_tensor), dim = 0)
        
        x1 = selected_data.clone().unsqueeze(0)
        x2 = dataset_mapped.clone().unsqueeze(1)
        
        with torch.no_grad():      
            similarity = F.cosine_similarity(x1, x2, dim=2)
            remaining_indices = torch.where(similarity <= threshold)[0]
            dataset_mapped = dataset_mapped[remaining_indices]

        # remaining_indices = torch.where(similarity <= threshold)[0].cpu().detach().numpy()
        # dataset_mapped = dataset_mapped[remaining_indices]
        
        selected_datas.append(selected_data.cpu().detach().numpy())
        
        torch.cuda.empty_cache()
        # del similarity, x1, x2
        del similarity, remaining_indices, x1, x2
        torch.cuda.empty_cache()
            
    sampling_ratio = len(selected_datas) / length
    return sampling_ratio


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
                                               pin_memory = False)
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
                    latent = model.encoder(inputs)
                    outputs = model.decoder(latent)
                    loss = criterion(outputs, target_data)
                    # outputs = model(inputs)
                    # loss = criterion(outputs, target_data)
                    
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                running_loss += loss.item() * inputs.size(0)

            scheduler.step(running_loss / len(dataset))
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

                latent = model.encoder(inputs)
                outputs = model.decoder(latent)
                loss = criterion(outputs, target_data)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            # print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(input_data)}")   
                

def learning_rate_scheduler(epoch):
        
    return 0.1 ** (epoch//10)


class RandomRotatePatches:
    def __init__(self, num_patches, max_angle=180):
        self.num_patches = num_patches
        self.max_angle = max_angle

    def __call__(self, patches):

        if isinstance(patches, torch.Tensor):
            patches = patches.cpu().numpy()
        # 패치를 2D 배열로 변환
        batch_size = patches.shape[0] // (self.num_patches[0] * self.num_patches[1])
        patches_2d_array = patches.reshape(batch_size, self.num_patches[0], self.num_patches[1], -1)
        
        # 랜덤 각도로 회전
        rotated_patches_list = []
        for batch in range(batch_size):
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            rotated_batch = rotate(patches_2d_array[batch], angle, axes=(0, 1), reshape=False)
            rotated_patches_list.append(rotated_batch)
        
        rotated_patches_array = np.stack(rotated_patches_list, axis=0)
        
        # 회전된 2D 배열을 다시 패치로 변환
        rotated_patches = rotated_patches_array.reshape(batch_size * self.num_patches[0] * self.num_patches[1], -1)
        return torch.tensor(rotated_patches).to(device)