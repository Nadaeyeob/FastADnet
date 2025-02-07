import os
import random
import shutil
import click
import sys
import torch
import numpy as np

# Random Seed 전부 막아야함

@click.group(chain=True)
def main():
    pass

@main.command('fewshot')
@click.option('--input_folder_name', type=str)
@click.option('--dataset_path', type=str, help='Path to the MVTec AD dataset.')
@click.option('--output_folder_name', type=str, help='Name of the output folder.')
@click.option('--seed', type=int, default=0, help='Random seed for sampling.')
@click.option('--sample_ratio', type=float, default=0.1, help='Sampling ratio for few-shot learning.')
def sampling_few_shot_data(input_folder_name, dataset_path, output_folder_name, seed, sample_ratio):

    set_seed(seed)
    mvtec_dataset = os.path.join(dataset_path, input_folder_name)
    if not os.path.exists(mvtec_dataset):
        print(f"Error: Data folder is not found.")
        sys.exit(1)

    output_path = os.path.join(dataset_path, output_folder_name)
    os.makedirs(output_path, exist_ok=True)

    class_names = list_folders(mvtec_dataset)
    for class_name in class_names:
        good_path = os.path.join(mvtec_dataset, class_name, 'train', 'good')
        output_good_path = os.path.join(output_path, class_name, 'train', 'good')

        os.makedirs(output_good_path, exist_ok=True)
        image_files = [f for f in os.listdir(good_path) if os.path.isfile(os.path.join(good_path, f))]
        # num_samples = max(1, int(len(image_files) * sample_ratio))
        num_samples = max(1, int((sample_ratio)))
        sampled_images = random.sample(image_files, num_samples)

        for image in sampled_images:
            src_path = os.path.join(good_path, image)
            dest_path = os.path.join(output_good_path, image)
            shutil.copy(src_path, dest_path)

        bad_path = os.path.join(mvtec_dataset, class_name, 'test')
        output_bad_path = os.path.join(output_path, class_name, 'test')

        try:
            shutil.copytree(bad_path, output_bad_path, copy_function=shutil.copy)
        except FileExistsError:
            shutil.rmtree(output_bad_path)
            shutil.copytree(bad_path, output_bad_path)

        gt_path = os.path.join(mvtec_dataset, class_name, 'ground_truth')
        output_gt_path = os.path.join(output_path, class_name, 'ground_truth')

        try:
            shutil.copytree(gt_path, output_gt_path, copy_function=shutil.copy)
        except FileExistsError:
            shutil.rmtree(output_gt_path)
            shutil.copytree(gt_path, output_gt_path)

def list_folders(dataset_path):
    return [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    main()
