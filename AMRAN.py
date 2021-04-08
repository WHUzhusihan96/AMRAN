from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import time
import data_loader
import ResNet as models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Transfer Framework')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=5)
parser.add_argument('--l2_decay', type=float, default=5e-4)
parser.add_argument('--mu', type=float, default=0)
parser.add_argument('--root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/")
parser.add_argument('--source_dir', type=str, default="UCM")
parser.add_argument('--test_dir', type=str, default="RSSCN7")
# RSTL
# UCM WHU AID RSSCN7
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

source_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)


def train(epoch, model):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE))

    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.Inception.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if args.cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)

        optimizer.zero_grad()
        # print(data_source.shape)
        s_output, mmd_loss = model(data_source, data_target, label_source, args.mu)
        cls_loss = criterion(s_output, label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlabel_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, i * len(data_source), len_source_dataset,
                100. * i / len_source_loader, loss.item(), cls_loss.item(), mmd_loss.item()))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in target_test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            s_output, _ = model(data, data, target, args.mu)
            test_loss += criterion(s_output, target)# sum up batch loss
            pred = torch.max(s_output, 1)[1]  # get the index of the max log-probability
            correct += torch.sum(pred == target)
        test_loss /= len_target_dataset
        print(args.test_dir, '  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
    return correct


if __name__ == '__main__':
    model = models.AMRANNet(num_classes=6)
    print(args)
    # print(model)
    # print(args.source_dir, "->", args.test_dir)
    path = './record/' + args.source_dir + '->' + args.test_dir + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    if args.cuda:
        model.cuda()
    correct = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
            best_model = os.path.join(path, 'best_dict.pkl')
            torch.save(model.state_dict(), best_model)
        print("%s max correct:" % args.test_dir, correct.item())
        print(args.source_dir, "to", args.test_dir)
    print(args)
