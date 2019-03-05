import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 10
num_epochs = 100
batch_size = 100
learning_rate = 0.0001
DIM = 32


# Download and construct CIFAR-10 dataset. Each image is 3x32x32
transform_train = transforms.Compose([
    #transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.abs(torch.randn(1)).item(),
            contrast=0.1*torch.abs(torch.randn(1)).item(),
            saturation=0.1*torch.abs(torch.randn(1)).item(),
            hue=0.1*torch.abs(torch.randn(1)).item()),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./', train=True, transform=transform_train, download=True)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8)

# Convolutional neural network (eight convolutional layers)
class cnn(nn.Module):
    def __init__(self, num_classes=10):
        super(cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2), # 33x33
            nn.ReLU(),
            nn.BatchNorm2d(64))                                   
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), # 34x34
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 17x17
            nn.Dropout2d(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), # 18x18
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), # 19x19
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 9x9
            nn.Dropout2d(0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), #10x10
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # 8x8
            nn.ReLU(),
            nn.Dropout2d(0.2))
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # 6x6
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # 4x4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d())    
        self.fc1 = nn.Linear(4*4*64, 500)
        self.fc2 = nn.Linear(500, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = cnn(num_classes)
if torch.cuda.device_count() > 1:
  print("There are", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
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

