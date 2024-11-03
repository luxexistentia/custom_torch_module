import time
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
from tqdm.auto import tqdm
from custom_torch_module import engine
import shutil
import random
import os

def freeze_model(model, trainable_part, classifier):
    print(f"[info] Freezing the model(trainable part {trainable_part})...")
    
    for param in model.parameters():
        param.requires_grad = True
    
    len_params = len(list(model.parameters()))
    
    for i, param in enumerate(model.parameters()):
        if (1 - trainable_part) > (i / len_params):
            param.requires_grad = False
    
    for param in classifier.parameters():
        param.requires_grad = True
    print(f"[info] Successfully Froze the model!(trainable part {trainable_part})\n")

    return model

def build_loader(train_dir, test_dir, train_transform, test_transform, custom_dataset=None, batch_size=32, device="cpu", dataset_size=1):
    if custom_dataset:
        train_data = custom_dataset(train_dir, train_transform)
        test_data = custom_dataset(test_dir, test_transform)
    else:
        train_data = datasets.ImageFolder(train_dir, train_transform, allow_empty=True)
        test_data = datasets.ImageFolder(test_dir, test_transform, allow_empty=True)

    if dataset_size < 1:
        data_len = int(len(train_data) * dataset_size)
        
        train_rand_idx = torch.randint(len(train_data), (data_len,))
        
        train_data = [train_data[idx] for idx in train_rand_idx.tolist()]
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def measure_time(test_func, num_imgs, input_data=None, trial_num=5):
    elapsed_time = 0
    for i in tqdm(range(trial_num)):
        start_time = time.time()
        
        if input_data != None:
            test_func(input_data)
        else:
            test_func()
        
        end_time = time.time()
        
        elapsed_time += end_time - start_time

    imgs_per_sec = trial_num * num_imgs / elapsed_time

    print(f"{trial_num} Trials has done")
    print(f"Total number of the Images : {num_imgs * trial_num}")
    print(f"Elapsed time : {elapsed_time:.2f} seconds")
    print(f"Imgs / sec : {imgs_per_sec:.2f} fps")

def build_transform(img_size, is_data_aug=False, interpolation="bicubic"):
    return timm.data.create_transform(input_size=img_size, interpolation=interpolation, is_training=is_data_aug)

def split_dataset(origin_dir, train_dir, test_dir, test_percent=0.2):
    test_path_list = [f for f in os.listdir(origin_dir) if os.path.isfile(os.path.join(origin_dir, f))]
    total_num = len(test_path_list)
    
    print(f"[info] Total Files : {total_num} files")
    if total_num == 0:
        print(f"[info] No file was found. Please check the directories")
        return
    
    test_path_list = random.sample(test_path_list, k=round(len(test_path_list)*test_percent))

    for file_path in tqdm(test_path_list):
        shutil.move(os.path.join(origin_dir, file_path), os.path.join(test_dir, file_path))
    print(f"[info] {len(test_path_list)} Files({100*len(test_path_list)/total_num:.2f}%) has been moved to test directory({test_dir})")
    
    train_path_list = [f for f in os.listdir(origin_dir) if os.path.isfile(os.path.join(origin_dir, f))]
    for file_path in tqdm(train_path_list):
        shutil.move(os.path.join(origin_dir, file_path), os.path.join(train_dir, file_path))
    print(f"[info] {len(train_path_list)} Files({100*len(train_path_list)/total_num:.2f}%) has been moved to train directory({train_dir})")
        
def merge_dataset(target_dir, train_dir, test_dir):
    test_path_list = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    train_path_list = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]

    test_len = len(test_path_list)
    train_len = len(train_path_list)
    total_num = test_len + train_len
    print(f"[info] Total Files : {total_num} files")
    if total_num == 0:
        print(f"[info] No file was found. Please check the directories")
        return

    for file_path in tqdm(test_path_list):
        shutil.move(os.path.join(test_dir, file_path), os.path.join(target_dir, file_path))
    print(f"[info] {test_len} Files({100*test_len/total_num:.2f}%) has been moved to target directory({target_dir})")
    
    for file_path in tqdm(train_path_list):
        shutil.move(os.path.join(train_dir, file_path), os.path.join(target_dir, file_path))
    print(f"[info] {train_len} Files({100*train_len/total_num:.2f}%) has been moved to target directory({target_dir})")