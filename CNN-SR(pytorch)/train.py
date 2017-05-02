import torch
import torch.optim as optim
import torch.nn as nn
from __future__ import print_function
from torch.autograd import Variable
from model import Net

def main():
    pass

def train(trainDataLoader, model, criterion, optimizer, epoch):

    for iter, trainDataBatch in enumerate(trainDataLoader, 1):
        lrIMG, hrIMG = Variable(trainDataBatch[0]), Variable(trainDataBatch[1], requires_grad=False)
        lrIMG, hrIMG = lrIMG.cuda(), hrIMG.cude()

        loss = criterion(model(lrIMG), hrIMG)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("[epoch %2.4f] loss %.4f\t " % (epoch + float(iter) / len(trainDataLoader), loss.data[0]))


if __name__ == "__main__":
    main()