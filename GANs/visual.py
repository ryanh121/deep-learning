import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Hyper-parameters
# num_classes = 10
# num_epochs = 200
batch_size = 128
# learning_rate = 0.0001
# DIM = 32

class discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196,32,32)),
            nn.LeakyReLU())                                   
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((196,16,16)),               
            nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1), 
            nn.LayerNorm((196,16,16)),
            nn.LeakyReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((196,8,8)),                
            nn.LeakyReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196,8,8)),
            nn.LeakyReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1), 
            nn.LayerNorm((196,8,8)),
            nn.LeakyReLU())
        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196,8,8)),
            nn.LeakyReLU())
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((196,4,4)),
            nn.LeakyReLU())    
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)
        
    def forward(self, x, extract_features=0):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if(extract_features==4):
            x = F.max_pool2d(x,8,8)
            # x = x.view(-1, no_of_hidden_units)
            x = x.reshape(x.size(0), -1)
            return x
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        if(extract_features==8):
            x = F.max_pool2d(x,4,4)
            # x = x.view(-1, no_of_hidden_units)
            x = x.reshape(x.size(0), -1)
            return x
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        fc1 = self.fc1(x)
        fc10 = self.fc10(x)
        return fc1,fc10

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

# model = torch.load('/mnt/d/IE 534/hw4/from-bw/model.ckpt')
model = torch.load('cifar10.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

## save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

# Get the output from the fc10 layer and report the classification accuracy.
_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)

# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)


# Synthetic Images Maximizing Classification Output
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001

models = [torch.load('cifar10.model'),torch.load('discriminator.model')]
# models = [torch.load('cifar10.model'),torch.load('tempD.model')]
names =['without_gen','with_gen']

for model,name in zip(models,names):
    model.cuda()
    model.eval()
    for i in range(200):
        _, output = model(X)

        loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
        print(i,accuracy,-loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples)
    plt.savefig('visualization/max_class_{}.png'.format(name), bbox_inches='tight')
    plt.close(fig)

# Synthetic Features Maximizing Features at Various Layers
# batch_size = 196
models = [torch.load('cifar10.model'),torch.load('tempD.model')]
names =['without_gen','with_gen']
for model,name in zip(models,names):
    for extract_features in [4,8]:
        X = X_batch.mean(dim=0)
        X = X.repeat(batch_size,1,1,1)

        Y = torch.arange(batch_size).type(torch.int64)
        Y = Variable(Y).cuda()

        lr = 0.1
        weight_decay = 0.001
        for i in range(200):
            output = model(X,extract_features)

            loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
            gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                    grad_outputs=torch.ones(loss.size()).cuda(),
                                    create_graph=True, retain_graph=False, only_inputs=True)[0]

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
            print(i,accuracy,-loss)

            X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
            X[X>1.0] = 1.0
            X[X<-1.0] = -1.0

        ## save new images
        samples = X.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)

        fig = plot(samples[0:100])
        plt.savefig('visualization/max_features_at_layer{}_{}.png'.format(extract_features,name), bbox_inches='tight')
        plt.close(fig)

