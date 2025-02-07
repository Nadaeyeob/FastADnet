#%%
import numpy as np
score_identical = np.load("/home/lab/patch_core_/score_identical.npy")
score_sampling = np.load("/home/lab/patch_core_/score_sampling.npy")

diff = np.abs(score_identical - score_sampling)
num = np.argmax(diff)

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

d = "capsule_test_crack_008.png"
image_path1 = f"/home/lab/patch_core_/results/MVTecAD_Results/debug/segmentation_images_diff_identical/mvtec_capsule/{d}"
image_path2 = f"/home/lab/patch_core_/results/MVTecAD_Results/debug/segmentation_images_diff_sampling/mvtec_capsule/{d}"
image_path3 = f"/home/lab/patch_core_/results/MVTecAD_Results/debug/segmentation_images_recon_identical/mvtec_capsule/{d}"
image_path4 = f"/home/lab/patch_core_/results/MVTecAD_Results/debug/segmentation_images_recon_sampling/mvtec_capsule/{d}"
image_path5 = f"/home/lab/patch_core_/results/MVTecAD_Results/debug/segmentation_images_original/mvtec_capsule/{d}"
image_path6 = f"/home/lab/patch_core_/results/MVTecAD_Results/debug/segmentation_images_identical/mvtec_capsule/{d}"
image_path7 = f"/home/lab/patch_core_/results/MVTecAD_Results/debug/segmentation_images_sampling/mvtec_capsule/{d}"

image1 = mpimg.imread(image_path1)
image2 = mpimg.imread(image_path2)
image3 = mpimg.imread(image_path3)
image4 = mpimg.imread(image_path4)
image5 = mpimg.imread(image_path5)
image6 = mpimg.imread(image_path6)
image7 = mpimg.imread(image_path7)
fig, axs = plt.subplots(7, 1, figsize=(15, 20))

axs[0].imshow(image3)
axs[0].set_title('identical')
axs[0].axis('off')

axs[1].imshow(image4)
axs[1].set_title('sampling')
axs[1].axis('off')

axs[2].imshow(image5)
axs[2].set_title('original')
axs[2].axis('off')

axs[3].imshow(image1)
axs[3].set_title('diff_identical')
axs[3].axis('off')

axs[4].imshow(image2)
axs[4].set_title('diff_sampling')
axs[4].axis('off')

axs[5].imshow(image6)
axs[5].set_title('diff_identical_not_abs')
axs[5].axis('off')

axs[6].imshow(image7)
axs[6].set_title('diff_sampling_not_abs')
axs[6].axis('off')

plt.tight_layout()  # subplot 간 간격 조절
plt.show()
print('finish')
# %%
import matplotlib.pyplot as plt
import numpy as np
segmentation_identical = np.load("/home/lab/patch_core_/segmentation_diff_identical.npy")
segmentation_sampling = np.load("/home/lab/patch_core_/segmentation_diff_sampling.npy")
segmentation_recon_identical = np.load("/home/lab/patch_core_/segmentation_recon_identical.npy")
segmentation_recon_sampling = np.load("/home/lab/patch_core_/segmentation_recon_sampling.npy")
segmentation_original = np.load("/home/lab/patch_core_/segmentation_recon_original.npy")
segmentation_train = np.load("/home/lab/patch_core_/segmentation_train.npy")
i = 0
threshold = 0.95
threshold_train = 0.8
segmentation_di = (segmentation_original[i] - segmentation_recon_identical[i])
segmentation_ds = (segmentation_original[i] - segmentation_recon_sampling[i])
segmentation_dt = (segmentation_original[i] - segmentation_train[1])
segmentation_i = segmentation_recon_identical[i] >= threshold
segmentation_s = segmentation_recon_sampling[i]  >= threshold
segmentation_o = segmentation_original[i]  >= threshold
segmentation_t = segmentation_train # 0~4
fig, axes = plt.subplots(2, 5)
axes[0][0].imshow(segmentation_di, cmap='gray')
axes[0][0].set_title('Diff_I')
axes[0][1].imshow(segmentation_ds, cmap='gray')
axes[0][1].set_title('Diff_S')
axes[0][2].imshow(segmentation_i, cmap='gray')
axes[0][2].set_title('I')
axes[0][3].imshow(segmentation_s, cmap='gray')
axes[0][3].set_title('S')
axes[0][4].imshow(segmentation_o, cmap='gray')
axes[0][4].set_title('O')
axes[1][0].imshow(segmentation_t[0], cmap='gray')
axes[1][0].set_title('T1')
axes[1][1].imshow(segmentation_t[1], cmap='gray')
axes[1][1].set_title('T2')
axes[1][2].imshow(segmentation_t[2], cmap='gray')
axes[1][2].set_title('T3')
axes[1][3].imshow(segmentation_t[3], cmap='gray')
axes[1][3].set_title('T4')
axes[1][4].imshow(segmentation_dt, cmap='gray')
axes[1][4].set_title('test')
for row in axes:
    for ax in row:
        ax.axis('off')
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
features_identical = np.load("/home/lab/patch_core_/features_identical.npy")
features_sampling = np.load("/home/lab/patch_core_/features_sampling.npy")
features_original = np.load("/home/lab/patch_core_/features_original.npy")
features_train = np.load("/home/lab/patch_core_/features_train.npy")

j = 1
features_identical = features_identical.reshape(-1, 784, 1024)[j]
features_sampling = features_sampling.reshape(-1, 784, 1024)[j]
features_original = features_original.reshape(-1, 784, 1024)[j]
features_train = features_train.reshape(-1, 784, 1024)[2]

diff_identical = features_identical - features_original
diff_sampling = features_sampling - features_original
diff_i_o = features_original - features_identical
diff_s_o = features_original - features_sampling

diff_i = np.sum(np.abs(diff_identical))
diff_s = np.sum(np.abs(diff_sampling))
diff_i_o = np.sum(np.abs(diff_i_o))
diff_s_o = np.sum(np.abs(diff_s_o))
print(diff_i, diff_s, diff_i_o, diff_s_o)

diff_new = features_original - features_train
# scaler = PCA(n_components=2)
# scaler.fit(features_train)
scaler = umap.UMAP()
scaler.fit(features_train)

identical_transformed = scaler.transform(features_identical)
sampling_transformed = scaler.transform(features_sampling)
original_transformed = scaler.transform(features_original)
train_transformed = scaler.transform(features_train)
diff_identical_transformed = scaler.transform(diff_identical)
diff_sampling_transformed = scaler.transform(diff_sampling)
diff_new_transformed = scaler.transform(diff_new)

fig, axes = plt.subplots(1, 7, figsize=(15, 6))
for i, data in enumerate([(train_transformed, 'Train'),
                        (identical_transformed, 'Identical'),
                        (sampling_transformed, 'Sampling'),
                        (original_transformed, 'Original'),
                        (diff_identical_transformed, 'diff_i'),
                        (diff_sampling_transformed, 'diff_s'),
                        (diff_new_transformed, 'diff_n')]):
    X, label = data
    ax = axes[i]
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.5)
    ax.set_title(label)
    # ax.set_xlim(-1, 0)
    # ax.set_ylim(-1, 1)

plt.tight_layout()
plt.show()
# %%
score_i = np.load("/home/lab/patch_core_/score_sampling.npy")
score_s = np.load("/home/lab/patch_core_/score_identical.npy")
print(np.sum(score_i - score_s))
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
features_identical = np.load("/home/lab/patch_core_/featrues_identical.npy")
features_sampling = np.load("/home/lab/patch_core_/featrues_identical.npy")
features_original = np.load("/home/lab/patch_core_/featrues_identical.npy")
features_train = np.load("/home/lab/patch_core_/featrues_identical.npy")

j = 8
features_identical = features_identical.reshape(-1, 784, 1024)[j]
features_sampling = features_sampling.reshape(-1, 784, 1024)[j]
features_original = features_original.reshape(-1, 784, 1024)[j]
features_train = features_train.reshape(-1, 784, 1024)[0]

diff_identical = features_original - features_sampling
diff_sampling = features_original - features_sampling

diff_identical.reshape(-1,1)
diff_sampling.reshape(-1,1)

np.sum(diff_sampling)
# %%
import torch
import numpy as np
from sklearn.neighbors import KernelDensity

def estimate_density(data, bandwidth=0.1):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(data)
    return torch.exp(torch.tensor(kde.score_samples(data)))


patches = np.load("/home/lab/patch_core_/features_manyshot_train.npy")
density = estimate_density(patches)

sorted_indices = torch.argsort(density, descending=True)

threshold_density = 0.01 
unique_patches = sorted_indices[density[sorted_indices] > threshold_density]

print("밀도를 기반으로 유니크한 Patch 개수:", len(unique_patches))
print(f"Density : len(unique_patches)/len(patches) = {len(unique_patches)/len(patches)}")

# %%
import torch
import itertools
import numpy as np
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = np.load("/home/lab/patch_core_/features_manyshot_train.npy")
dataset = torch.tensor(dataset, device = device)
mapper = torch.nn.Linear(
    dataset.shape[1], 128, bias=False
)
_ = mapper.to(device)
dataset = mapper(dataset)

def cosine_similarity(vec1, vec2):
    similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(1), dim=2)
    return similarity

threshold = 0.99
selected_datas = []
i = 0
length = len(dataset)
print(length)
while len(dataset) > 0:
    selected_data_index = random.randint(0, len(dataset) - 1)
    selected_data = dataset[selected_data_index]
    
    similarities = cosine_similarity(selected_data, dataset)
    
    remaining_indices = torch.where(similarities <= threshold)[0]
    dataset = dataset[remaining_indices]
    
    selected_datas.append(selected_data)
    torch.cuda.empty_cache()
    
print(f"length of datasets = {len(selected_datas)}")
print(f"Density = {len(selected_datas) / length}")
torch.cuda.empty_cache()

# %%
