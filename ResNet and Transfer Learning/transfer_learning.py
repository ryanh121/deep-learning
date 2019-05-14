import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
# import copy

num_classes = 100
num_epochs = 100
batch_size = 100
learning_rate = 0.0001
DIM = 224

# Download and construct CIFAR-100 dataset. Each image is 3x32x32
transform_train = transforms.Compose([
    #transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.Resize(224, interpolation=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR100(
    root='/projects/training/bawc/CIFAR100/Dataset/', train=True, transform=transform_train, download=True)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(
    root='/projects/training/bawc/CIFAR100/Dataset/', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8)

# def resnet18(pretrained = True):
#     model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     }
#     model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])
#     if pretrained:
#         model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet18'], model_dir ='./'))
#     return model

# set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
# model = resnet18()
model = models.resnet18(pretrained=True)
# layer4 = copy.deepcopy(model.layer4)
for i,child in enumerate(model.children()):
    if i == 7:
        break
    for param in child.parameters():
        param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
# model.layer4 = layer4
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if torch.cuda.device_count() > 1:
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
print("There are", torch.cuda.device_count(), "GPU(s)!")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.layer4.parameters())+list(model.fc.parameters()), lr=learning_rate)

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
            with open('./trans_loss', 'a') as f:
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
            with open('./trans_accuracy', 'a') as f:
                print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total), file=f)
            
# Save the model checkpoint
torch.save(model.state_dict(), 'model_trans.ckpt')
