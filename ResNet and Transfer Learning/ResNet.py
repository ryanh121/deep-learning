import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 100
num_epochs = 100
batch_size = 100
learning_rate = 0.0001
DIM = 32

# Download and construct CIFAR-100 dataset. Each image is 3x32x32
transform_train = transforms.Compose([
    #transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR100(
    root='/projects/training/bawc/CIFAR100/Dataset/', train=True, transform=transform_train, download=True)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(
    root='/projects/training/bawc/CIFAR100/Dataset/', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8)

class basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(basicblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size, stride=1, padding=padding)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels))
    
    def forward(self, x):
        out = self.conv1(x)
        #print('out: {}'.format(out.shape))
        out = self.batch1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #print('out: {}'.format(out.shape))
        out = self.batch2(out)
        if self.downsample:
            x = self.downsample(x)
            #print('downsample x: {}'.format(x.shape))
        out += x
        out = self.relu(out)
        return out

# Convolutional neural network (eight convolutional layers)
class resnet(nn.Module):
    def __init__(self, block, num_classes=num_classes):
        super(resnet, self).__init__()
        self.blockchannels = 32
        self.block = block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2))                                   
        self.conv2_x = self.blocks(num_blocks=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3_x = self.blocks(num_blocks=4, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4_x = self.blocks(num_blocks=4, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv5_x = self.blocks(num_blocks=2, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1024, num_classes)

    def blocks(self, num_blocks, out_channels, kernel_size, stride, padding):
        result = [self.block(self.blockchannels, out_channels, kernel_size, stride, padding)]
        for i in range(1,num_blocks):
            result.append(self.block(out_channels, out_channels, kernel_size, 1, padding))
        self.blockchannels = out_channels
        return nn.Sequential(*result)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        #print('forward x.shape: {}', x.shape)
        x = self.fc1(x)
        return x

model = resnet(basicblock)

if torch.cuda.device_count() > 1:
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
print("There are", torch.cuda.device_count(), "GPU(s)!")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model, and test the model every 5 epochs 
total_step = len(trainloader)
for epoch_index, epoch in enumerate(range(num_epochs)):
    # Train the model
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        if(epoch > 6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            with open('./loss', 'a') as f:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()), file = f)

    # Test the model
    if (epoch_index+1) % 5 == 0:
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance, dropout uses p instead of randomly setting some parameters to be 0)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
            with open('./accuracy', 'a') as f:
                print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total), file=f)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

