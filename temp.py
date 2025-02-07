import numpy as np
from scipy.ndimage import rotate

class RandomRotatePatches:
    def __init__(self, num_patches, max_angle=180):
        self.num_patches = num_patches
        self.max_angle = max_angle

    def __call__(self, patches):
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
        return rotated_patches

# 예시 사용법
original_patches = np.random.rand(2 * 784, 1024)  # (batch_size * number of patch, 1024)
num_patches = (28, 28)  # 256 / 16 = 16

# 데이터 증강 객체 생성
random_rotate_patches = RandomRotatePatches(num_patches)

# 데이터 증강 적용
augmented_patches = random_rotate_patches(original_patches)
print(augmented_patches.shape)  # (batch_size * number of patch, 1024)

