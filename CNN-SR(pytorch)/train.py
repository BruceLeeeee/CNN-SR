from __future__ import print_function
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import Net
from data import DataSet


def main():
    data_set = DataSet("data/train5/")
    train_data_loader = DataLoader(dataset=data_set, num_workers=4, batch_size=64, shuffle=True)
    model = Net()
    criterion = nn.MSELoss(size_average=False)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(20):
        train(train_data_loader, model, criterion, optimizer, epoch)
        save_checkpoint(model, epoch)

    print("train done--------->")


def train(train_data_loader, model, criterion, optimizer, epoch):
    for iter, train_data_batch in enumerate(train_data_loader, 1):
        lr_img, hr_img = Variable(train_data_batch[0]), Variable(train_data_batch[1], requires_grad=False)
        if torch.cuda.is_available():
            lr_img, hr_img = lr_img.cuda(), hr_img.cuda()

        loss = criterion(model(lr_img), hr_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("[epoch %2.4f] loss %.4f\t " % (epoch + float(iter) / len(train_data_loader), loss.data[0]))


def save_checkpoint(model, epoch):
    model_dir = "model/" + "model_epoch_{}.pth".format(epoch)
    state_dict = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state_dict, model_dir)
    print("checkpoint saved[epoch:{}]--------->".format(epoch))


if __name__ == "__main__":
    main()
