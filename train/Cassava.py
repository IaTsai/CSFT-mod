# train/Cassava.py

import os
import time
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.cassava_dataset import CassavaDataset
from models.CSFT import CrossScaleFusionTransformer
import numpy as np
from tqdm import tqdm


print("Using device:", torch.cuda.current_device())   # 印出 index
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

def train(model, record, train_loader, criterion, optimizer, device):
    print("iii-DEBUG record:", record)

    record.setdefault('epoch', 0)
    model.train()
    total, batch_acc, batch_loss = 0, 0, 0.
    #for images, labels in train_loader:
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {record['epoch']+1}")):
    ## 這裡的 batch_idx 自動就是 0, 1, 2, ... 的整數
        print(f"Batch {batch_idx+1}/{len(train_loader)}")
        images, labels = images.to(device), labels.to(device)
        output = model(images, images, images)  # 簡化處理：同圖像輸入 3 次
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == labels).sum()
        total += len(labels)
        batch_acc += acc
        batch_loss += loss
    epoch_loss = batch_loss.item() / len(train_loader)
    epoch_acc = 100 * batch_acc.item() / total
    record["Lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
    record["Loss_train"].append(epoch_loss)
    record["Acc_train"].append(epoch_acc)
    print("Train_loss: {:.4f} Train_acc: {:.4f}%".format(epoch_loss, epoch_acc))

def test(model, record, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        total, batch_acc, batch_loss = 0, 0, 0.
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images, images, images)  # 同上：三路輸入
            loss = criterion(output, labels)
            acc = (output.argmax(dim=1) == labels).sum()
            total += len(labels)
            batch_acc += acc
            batch_loss += loss
        epoch_loss = batch_loss.item() / len(test_loader)
        epoch_acc = 100 * batch_acc.item() / total
        record["Loss_test"].append(epoch_loss)
        record["Acc_test"].append(epoch_acc)
        print("Test_loss: {:.4f} Test_acc: {:.4f}%".format(epoch_loss, epoch_acc))
        return epoch_acc

def start(args, record, fold):
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # Dataset & Dataloader
    dataset = CassavaDataset(
        csv_file="/ssd5/ia313553058/DL_林彥宇/CSFT-cassava/CSFT-mod/datasets/cassava-leaf-disease-classification/train.csv",
        root_dir="/ssd5/ia313553058/DL_林彥宇/CSFT-cassava/CSFT-mod/datasets/cassava-leaf-disease-classification/train_images/",
        transform=transform
    )

    # 80/20 split
    total_size = len(dataset)
    split = int(0.8 * total_size)
    train_set, test_set = torch.utils.data.random_split(dataset, [split, total_size - split])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(set(pd.read_csv("/ssd5/ia313553058/DL_林彥宇/CSFT-cassava/CSFT-mod/datasets/cassava-leaf-disease-classification/train.csv")['label']))
    model = CrossScaleFusionTransformer(args.img_size, num_classes)
    
    if args.train:
        model.load_from(np.load(args.pretrained_dir))
    else:
        directory = args.test_model or os.path.join("weights", args.dataset + ".pt")
        print(f"Loading model from: {directory}")
        model.load_state_dict(torch.load(directory))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    cudnn.benchmark = True

    best = 0.0
    for epoch in range(args.epochs if args.train else 1):
        print(f"Fold{fold} Epoch {epoch+1}/{args.epochs}")
        s_time = time.perf_counter()
        if args.train:
            train(model, record, train_loader, criterion, optimizer, device)
        test_acc = test(model, record, test_loader, criterion, device)
        e_time = time.perf_counter()
        runtime = e_time - s_time
        print("Time: %d m %.3f s" % (runtime // 60, runtime % 60))
        record["Epoch"].append(epoch + 1)
        record["Time"].append(runtime)
        record["Batch_size"].append(args.batch_size)
        if best <= test_acc:
            best = test_acc
            if args.train:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"fold{fold}.pt"))
        record["Best"].append(best)
        print("Best: {:.4f}%".format(best))
        pd.DataFrame(record).to_csv(os.path.join(args.record_dir, f"fold{fold}.csv"), index=False)

def initial(args, times=1):
    for fold in range(times):
        record = {"Epoch": [], "Time": [], "Batch_size": [],
                  "Lr": [], "Loss_train": [], "Acc_train": [],
                  "Loss_test": [], "Acc_test": [], "Best": []} if args.train else {
                  "Epoch": [], "Time": [], "Batch_size": [], "Loss_test": [], "Acc_test": [], "Best": []}
        start(args, record, fold + 1)
