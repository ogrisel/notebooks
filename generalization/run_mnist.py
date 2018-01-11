from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='Study of generalization in MLPs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train-size', type=int, default=None,
                    help='size of the subsample used for training')
parser.add_argument('--test-size', type=int, default=None,
                    help='size of the subsample used for test evaluation')
parser.add_argument('--dropout', type=float, default=None,
                    help='dropout probability (no dropout by default)')
parser.add_argument('--mlp', action='store_true', default=False,
                    help='use an MLP instead of a ConvNet')
parser.add_argument('--hidden-dim', type=int, default=32,
                    help='dimension of the MLP hidden layers')
parser.add_argument('--depth', type=int, default=1,
                    help='number of hidden layers for the MLP')             
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

mnist_transformers = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def make_mnist_loader(train=True, subsample=None):
    dataset = datasets.MNIST('../data', train=train, download=True,
                             transform=mnist_transformers)
    if subsample is None:
        # Use the full training set
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    else:
        # Subsample a smaller training set at random
        mnist_loader = loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.train_size, shuffle=True, **loader_kwargs)
        small_mnist_data, small_mnist_labels = next(iter(mnist_loader))
        small_mnist_dataset = torch.utils.data.TensorDataset(
            small_mnist_data, small_mnist_labels)
        loader = torch.utils.data.DataLoader(
            small_mnist_dataset, batch_size=args.batch_size, shuffle=True,
            **loader_kwargs
        )
    return loader


train_loader = make_mnist_loader(train=True, subsample=args.train_size)
test_loader = make_mnist_loader(train=False, subsample=args.test_size)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        if args.dropout:
            self.conv2_drop = nn.Dropout2d(p=args.dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        if args.dropout:
            x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if args.dropout:
            x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class MLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden=(32,)):
        super(MLP, self).__init__()
        self.hidden_layers = layers = []
        for hidden_dim in hidden:
            layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for h in self.hidden_layers:
            x = F.relu(h(x))
        return F.log_softmax(self.output_linear(x))


if args.mlp:
    model = MLP(hidden=[args.hidden_dim] * args.depth)
else:
    model = ConvNet()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, cooldown=5,
                              verbose=True)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if isinstance(model, MLP):
            data = data.view(-1, 784)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:f}'
                  .format(epoch, batch_idx * len(data),
                          len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.data[0],
                          optimizer.param_groups[0]['lr']))


def evaluate():
    should_stop = False
    model.eval()

    for name, loader in [('train', train_loader), ('test', test_loader)]:
        loss = 0
        correct = 0
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if isinstance(model, MLP):
                data = data.view(-1, 784)
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss += F.nll_loss(output, target, size_average=False).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss /= len(loader.dataset)
        print('{} -- Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
              .format(name.ljust(5), loss, correct, len(loader.dataset),
                      100. * correct / len(loader.dataset)))
        if name == 'test':
            scheduler.step(loss)
            should_stop = should_stop or correct == len(loader.dataset)
    return should_stop or optimizer.param_groups[0]['lr'] < args.lr / 1e2


for epoch in range(1, args.epochs + 1):
    train(epoch)
    if evaluate():
        break
