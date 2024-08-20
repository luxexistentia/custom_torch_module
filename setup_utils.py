import time
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
from tqdm.auto import tqdm
from custom_torch_module import engine

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

def build_loader(train_dir, test_dir, train_transform, test_transform, batch_size=32, device="cpu"):
    train_data = datasets.ImageFolder(train_dir, train_transform)
    test_data = datasets.ImageFolder(test_dir, test_transform)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def measure_time(test_func, trial_device, num_imgs, trial_num=5):
    torch.cuda.empty_cache()
    oroginal_device = next(model.parameters()).device
    model = model.to(trial_device)
    
    elapsed_time = 0
    for i in tqdm(range(trial_num)):
        start_time = time.time()
        test_func()
        end_time = time.time()
        elapsed_time += end_time - start_time

    imgs_per_sec = trial_num * num_imgs / elapsed_time

    print(f"{trial_num} Trials(device {trial_device}) has done")
    print(f"Total number of the Images : {num_imgs * trial_num}")
    print(f"Elapsed time : {elapsed_time:.2f} seconds")
    print(f"Imgs / sec : {imgs_per_sec:.2f} fps")
    
    model = model.to(oroginal_device)

def build_transform(img_size, is_data_aug=False, interpolation="bicubic"):
    return timm.data.create_transform(input_size=img_size, interpolation=interpolation, is_training=is_data_aug)
