import time
import tqdm

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from training_sub import DataSet, EarlyStopping


def train_model(model, criterion, optimizer, num_epochs, args, device, run):

    early_stopping = EarlyStopping(patience=15, verbose=True, path=args.savedir_path + "/model_parameter.pth")

    data = np.load(args.training_validation_path)
    # data = np.memmap(args.training_validation_path, dtype=np.float32, mode="r", shape=(11676, 120, 112, 112))
    data = torch.from_numpy(data).float()
    label = [0] * len(data)
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, label, test_size=0.2, random_state=42, stratify=label
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        val_data, val_labels, test_size=0.25, random_state=42, stratify=train_labels
    )
    train_dataset = DataSet(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_mini_batch, shuffle=True)
    val_dataset = DataSet(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_mini_batch, shuffle=False)
    dataloader_dic = {"train": train_dataloader, "val": val_dataloader}

    train_loss_list = []
    val_loss_list = []
    start = time.time()
    for epoch in range(args.num_epoch):
        train_loss_num = 0
        val_loss_num = 0

        for phase in ["train", "val"]:
            dataloader = dataloader_dic[phase]
            if phase == "train":
                model.train()  # モデルを訓練モードに
            else:
                model.eval()

            for images, labels in tqdm.tqdm(dataloader):
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
                        train_loss_num += weighted_loss.item()
                    else:
                        val_loss_num += weighted_loss.item()

            if phase == "train":
                train_loss_list.append(train_loss_num)
                wandb.log({"train loss": train_loss_num})
            else:
                val_loss_list.append(val_loss_num)
                wandb.log({"value loss": train_loss_num})
            
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, val_loss_num))

        early_stopping(val_loss_num, model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break

    print((time.time() - start) / 60)
