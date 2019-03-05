# Data Augmentation
```
transform_train = transforms.Compose([
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
```
# Convolution Network Structure
```
Convolution layer 1: 3 input channels, 64 output channels, k = 4; s = 1; P = 2.
ReLU
Batch normalization
Convolution layer 2: 64 input channels, 64 output channels, k = 4; s = 1; P = 2.
ReLU
Max Pooling: s = 2, k = 2.
Dropout: p = 0.25
Convolution layer 3: 64 input channels, 64 output channels, k = 4; s = 1; P = 2.
ReLU
Batch normalization
Convolution layer 4: 64 input channels, 64 output channels, k = 4; s = 1; P = 2.
ReLU
Max Pooling
Dropout: p = 0.25
Convolution layer 5: 64 input channels, 64 output channels, k = 4; s = 1; P = 2.
ReLU
Batch normalization
Convolution layer 6: 64 input channels, 64 output channels, k = 3; s = 1; P = 0.
ReLU
Dropout: p = 0.25
Convolution layer 7: 64 input channels, 64 output channels, k = 3; s = 1; P = 0.
ReLU
Batch normalization
Convolution layer 8: 64 input channels, 64 output channels, k = 3; s = 1; P = 0.
ReLU
Batch normalization
Dropout: p = 0.5
Fully connected layer 1: 1024 input channels, 500 output channels.
Fully connected layer 2: 500 input channels, 10 output channels.
Linear Softmax function
```
# Test accuracies after every 5 epochs (totally 100 epochs)
```
Test Accuracy of the model on the 10000 test images: 63.8 %
Test Accuracy of the model on the 10000 test images: 72.39 %
Test Accuracy of the model on the 10000 test images: 76.57 %
Test Accuracy of the model on the 10000 test images: 78.09 %
Test Accuracy of the model on the 10000 test images: 79.75 %
Test Accuracy of the model on the 10000 test images: 81.19 %
Test Accuracy of the model on the 10000 test images: 81.63 %
Test Accuracy of the model on the 10000 test images: 82.18 %
Test Accuracy of the model on the 10000 test images: 82.98 %
Test Accuracy of the model on the 10000 test images: 83.54 %
Test Accuracy of the model on the 10000 test images: 84.44 %
Test Accuracy of the model on the 10000 test images: 84.75 %
Test Accuracy of the model on the 10000 test images: 84.78 %
Test Accuracy of the model on the 10000 test images: 85.19 %
Test Accuracy of the model on the 10000 test images: 85.58 %
Test Accuracy of the model on the 10000 test images: 85.65 %
Test Accuracy of the model on the 10000 test images: 85.76 %
Test Accuracy of the model on the 10000 test images: 85.87 %
Test Accuracy of the model on the 10000 test images: 86.11 %
Test Accuracy of the model on the 10000 test images: 86.34 %
```
