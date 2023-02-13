import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Train on GPU...")
else:
    print("Train on CPU...")

### Define Sqeeze and Excitation layer
class SE_layer(nn.Module): 
    def __init__(self, input_C=3, ratio=8):
        # Ratio r controls the amount of information that is used to calculate the self-attention weights
        super(SE_layer, self).__init__()
        self.input_C = input_C
        self.ratio = ratio
        self.squeeze = nn.AdaptiveAvgPool2d(1) # Output size 1*1
        self.excitation = nn.Sequential(
            # Reduce C channels to C/r channels --> ease computation
            nn.Linear(input_C, input_C//ratio, bias=False),
            nn.ReLU(inplace=True), 
            # Convert back to C channels
            nn.Linear(input_C//ratio, input_C, bias=False),
            nn.Sigmoid()
        ) 
    
    def forward(self, input_x): 
        N, C, H, W = input_x.shape
        x = self.squeeze(input_x) 
        x = x.view(N, C) 
        x = self.excitation(x) 
        x = x.view(N, C, 1, 1)
        x = input_x * x # Scale original channels by the weights
        return x 

### Define regulator -- convolutional LSTM
class ConvLSTM(nn.Module):
    def __init__(self, input_C, hidden_C):
        super(ConvLSTM, self).__init__()
        self.input_C = input_C # Number of channels of input tensor
        self.hidden_C = hidden_C # Number of channels of hidden state
        self.conv = nn.Sequential(
                        nn.Conv2d(input_C+hidden_C, 
                                  4*hidden_C, 
                                  kernel_size=3,
                                  padding=1,
                                  bias=True),
                        nn.BatchNorm2d(4*hidden_C),
                        nn.ReLU())

    def forward(self, input_x, cur_state):
        if cur_state is None:
            H_cur, C_cur = (torch.zeros(20, 10, 96, 96).to(device),
                            torch.zeros(20, 10, 96, 96).to(device))
        else:
            H_cur, C_cur = cur_state
        
        concat = torch.cat([input_x, H_cur], dim=1)  # concatenate along the channel axis
        concat_conv = self.conv(concat)
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_C, dim=1) # divide result tensor into 4 equal size chunks
        # Gating
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        C_next = f * C_cur + i * g # C = k+j = c*f+g*i
        H_next = o * torch.tanh(C_next)
        cur_state = (H_next, C_next) # update current state
        return H_next, cur_state

### Define RegNet
class RegNet(nn.Module):
    def __init__(self, input_C=3, hidden_C=10):
        super(RegNet, self).__init__()
        self.input_C = input_C
        self.hidden_C = hidden_C
        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(input_C, hidden_C, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(hidden_C),
                        nn.ReLU())
        self.ConvLSTM1 = ConvLSTM(hidden_C, hidden_C)
        self.conv2 = nn.Sequential(
                        nn.Conv2d(hidden_C+hidden_C, hidden_C, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(hidden_C),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(hidden_C, input_C, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(input_C))
        self.SElayer1 = SE_layer(input_C, ratio=8)
        self.conv4 = nn.Sequential(
                        nn.Conv2d(input_C, hidden_C, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(hidden_C),
                        nn.ReLU())
        self.ConvLSTM2 = ConvLSTM(hidden_C, hidden_C)
        self.conv5 = nn.Sequential(
                        nn.Conv2d(hidden_C+hidden_C, hidden_C, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(hidden_C),
                        nn.ReLU())
        self.conv6 = nn.Sequential(
                        nn.Conv2d(hidden_C, input_C, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(input_C))
        self.SElayer2 = SE_layer(input_C, ratio=8)
        self.conv7 = nn.Sequential(
                        nn.Conv2d(input_C, hidden_C, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(hidden_C),
                        nn.ReLU())
        self.ConvLSTM3 = ConvLSTM(hidden_C, hidden_C)
        self.conv8 = nn.Sequential(
                        nn.Conv2d(hidden_C+hidden_C, hidden_C, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(hidden_C),
                        nn.ReLU())
        self.conv9 = nn.Sequential(
                        nn.Conv2d(hidden_C, input_C, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(input_C))
        self.SElayer3 = SE_layer(input_C, ratio=8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(3*96*96, 10)
    
    def forward(self, input_x):
        # First block
        x1 = input_x
        x2 = self.conv1(x1)
        H, curstate = self.ConvLSTM1(x2, cur_state=None)
        concat = torch.cat([x2, H], dim=1)
        x3 = self.conv2(concat)
        x4 = self.conv3(x3)
        x5 = self.SElayer1(x4)
        out = x5+x1
        out = self.relu(out)
        # Second block
        x1 = out
        x2 = self.conv4(out)
        H, curstate = self.ConvLSTM2(x2, cur_state=curstate)
        concat = torch.cat([x2, H], dim=1)
        x3 = self.conv5(concat)
        x4 = self.conv6(x3)
        x5 = self.SElayer2(x4)
        out = x5+x1
        out = self.relu(out)
        # Third block
        x1 = out
        x2 = self.conv7(out)
        H, curstate = self.ConvLSTM3(x2, cur_state=curstate)
        concat = torch.cat([x2, H], dim=1)
        x3 = self.conv8(concat)
        x4 = self.conv9(x3)
        x5 = self.SElayer3(x4)
        out = x5+x1
        out = self.relu(out)
        # Fully connected layer
        out = out.view(input_x.size(0), -1)
        out = self.fc(out)
        return out

# Transformation
Transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load dataset
train = datasets.STL10('~/.pytorch/STL10', download=True, split='train', transform=Transform)
test = datasets.STL10('~/.pytorch/STL10', download=True, split='test', transform=Transform)

# Split training set into 80% training and 20% validation
train_num = 4000
val_num = 1000
train_set, val_set = random_split(train, [train_num, val_num])

# Create data loaders with batch size 20
train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=20, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=20, shuffle=False)

model = RegNet(input_C=3, hidden_C=10).to(device)
# Hyper-parameter values suggested by paper
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
max_epochs = 50
random_seed = 1
torch.manual_seed(random_seed)

# Names of classes
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

loss_list, acc_list = [], []
loss_list_val, acc_list_val = [], []
criterion = nn.CrossEntropyLoss()

for epoch in range(max_epochs):
    # training
    model.train()
    epoch_loss = 0.0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        y = model(data)
        #print(y)
        loss = criterion(y,labels)
        loss.backward()
        optimizer.step()

        predicted_label = torch.argmax(y, dim=1)
        correct += (predicted_label == labels).float().sum()
        epoch_loss += loss.item()
    avg_loss = epoch_loss/train_num
    avg_acc = correct/train_num
    loss_list.append(avg_loss)
    acc_list.append(avg_acc)
    
    # validation
    model.eval()
    with torch.no_grad():
        loss_val = 0.0
        correct_val = 0
        for batch_idx, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            y = model(data)
            loss = criterion(y,labels)
            predicted_label = torch.argmax(y,dim=1)
            correct_val += (predicted_label == labels).float().sum()
            loss_val += loss.item()
        avg_loss_val = loss_val/val_num
        avg_acc_val = correct_val/val_num
        loss_list_val.append(avg_loss_val)
        acc_list_val.append(avg_acc_val)
    print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f' % (epoch + 1, avg_loss, avg_acc, avg_loss_val, avg_acc_val))

# Plot the training loss and validation loss
epoch = range(1,51)
plt.plot(epoch,loss_list,label = "training loss")
plt.plot(epoch,loss_list_val,label = "validation loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training loss vs validation loss')
plt.legend()
plt.show()

## Plot the training accuracy and validation accuracy
plt.plot(epoch,torch.tensor(acc_list, device = 'cpu'),label = "training accuracy")
plt.plot(epoch,torch.tensor(acc_list_val, device = 'cpu'),label = "validation accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training accuracy vs validation accuracy')
plt.legend()
plt.show()

# Performance on the test set
true_labels = []
predictions = []
correct_test = 0
model.eval()
with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, label = data.to(device), labels.to(device)
        y = model(data)
        predicted_label = torch.argmax(y, dim=1)
        correct_test += (predicted_label == label).float().sum()
        predictions.append(predicted_label)
        true_labels.append(label)

print('Accuracy on the 8000 test images: %.2f %%' % (100 * correct_test / len(test)))
