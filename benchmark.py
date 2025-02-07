import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
import time
import numpy as np

class LastLayerToExtractReachedException(Exception):
    pass

class ForwardHook:
    def __init__(self, outputs, layer_name, last_layer):
        self.outputs = outputs
        self.layer_name = layer_name
        self.last_layer = last_layer

    def __call__(self, module, input, output):
        self.outputs[self.layer_name] = output
        if self.layer_name == self.last_layer:
            raise LastLayerToExtractReachedException

class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims: 
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)
    
class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)
    
class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)
    
def get_ae():
    return nn.Sequential(
        # nn.Linear(1024,512),
        # nn.LeakyReLU(),
        # nn.Linear(512,256),
        # nn.LeakyReLU(),
        # nn.Linear(256,128),
        # nn.LeakyReLU(),
        # nn.Linear(128,64),
        # nn.LeakyReLU(),

        # nn.Linear(64, 128),
        # nn.LeakyReLU(),
        # nn.Linear(128,256),
        # nn.LeakyReLU(),
        # nn.Linear(256,512),
        # nn.LeakyReLU(),
        # nn.Linear(512,1024),
        # nn.LeakyReLU(),

        nn.Linear(1024,1024),
        nn.LeakyReLU(),
    )


# Load the WideResNet-101 model
# resnet = models.wide_resnet101_2(pretrained=True)
resnet = models.wide_resnet50_2(pretrained=True)
# resnet = models.resnet50(pretrained=True)
# resnet = models.resnet34(pretrained=True)
# resnet = models.resnet18(pretrained=True)
# resnet = models.efficientnet_b0(pretrained=True)
# resnet = models.efficientnet_b1(pretrained=True)
# resnet = models.densenet121(pretrained=True)

resnet.eval()

# GPU usage
gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# Set model to half precision if CUDA is available
if torch.cuda.is_available():
    resnet.half()

# Specify the layers to extract from
# layers_to_extract_from = ["layer2"]
layers_to_extract_from = ["layer2", "layer3"]

# Create an instance of the NetworkFeatureAggregator
feature_aggregator = NetworkFeatureAggregator(resnet, layers_to_extract_from, device)
patch_maker = PatchMaker(patchsize=1, stride=1)
preprocessing = Preprocessing([512,1024],1024)
pooling = Aggregator(1024)
autoencoder = get_ae().to(device=device)
# Random input data
batch_size = 1
num_channels = 3
height, width = 224, 224
random_input = torch.rand(batch_size, num_channels, height, width)

# Set input to half precision if CUDA is available
if torch.cuda.is_available():
    random_input = random_input.half().cuda()
    autoencoder = autoencoder.half().cuda()

total_times = []
agg_times = []
patch_times = []
feature_times = []
encoder_times = []
iteration = 2000

with torch.inference_mode():
    for i in range(iteration):
        # Extract features
        random_input = torch.randn(1, 3, 224, 224, dtype=torch.float16 if gpu else torch.float32).to(device)
        start_time = time.time()
        
        features = feature_aggregator(random_input)
        
        agg_time = time.time() - start_time
        
        features = [features[layer] for layer in layers_to_extract_from]
        features = [patch_maker.patchify(x, return_spatial_info=True) for x in features]
        
        patch_time = time.time() - start_time
        
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = preprocessing(features)
        features = pooling(features)
        
        feature_time = time.time() - start_time
        
        features = autoencoder(features)
        
        encoder_time = time.time() - start_time
        
        total_time = time.time() - start_time
        
        
        agg_times.append(agg_time)
        patch_times.append(patch_time)
        feature_times.append(feature_time)
        encoder_times.append(encoder_time)
        total_times.append(total_time)

print(np.mean(agg_times[-1000:]))
print(np.mean(patch_times[-1000:]))
print(np.mean(feature_times[-1000:]))
print(np.mean(encoder_times[-1000:]))
print(np.mean(total_times[-1000:]))
