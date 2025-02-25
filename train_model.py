import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from training_sub import DataSet, EarlyStopping


def train_model(model, criterion, optimizer, num_epochs, args, device, run):

    early_stopping = EarlyStopping(patience=15, verbose=True, path=args.savedir + "/model_parameter.pth")

    data = np.load(args.training_validation_path)
    data = torch.from_numpy(data).float()
    label = [0] * len(data)
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, label, test_size=0.1, random_state=42, stratify=label
    )
    train_data, test_data, train_labels, test_labels = train_test_split(
        train_data, train_labels, test_size=0.05, random_state=42, stratify=train_labels
    )
    train_dataset = DataSet(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = DataSet(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    dataloader_dic = {"train": train_dataloader, "val": val_dataloader}

    loss_list = []
    start = time.time()
    for epoch in range(args.num_epoch):
        loss_num = 0

        for phase in ["train", "val"]:
            dataloader = dataloader_dic[phase]
            if phase == "train":
                model.train()  # モデルを訓練モードに
            else:
                model.eval()

            for images, labels in dataloader:
                images = images.view(-1, 1, 12, 112, 112)  # バッチサイズを維持したままチャンネル数を1に設定
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):

                    # モデルの出力を計算する
                    output = model(images.clone().to(device))

                    # 損失を計算する
                    loss = criterion(output.to("cpu"), images)
                    weighted_loss = torch.mean(loss)

                    # パラメータの更新
                    if phase == "train":
                        weighted_loss.backward()
                        optimizer.step()
                    else:
                        loss_num += weighted_loss.item()

        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss_num))
        loss_list.append(loss_num)

        early_stopping(loss_num, model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break

    print((time.time() - start) / 60)
