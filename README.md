### Abstract

This report is an attempt of implementing RegNet on the STL10 dataset[1]. 

RegNet is originally proposed in <RegNet:Self-regulated network for Image Classification>[2]. The main idea is to insert regulator modules (convolutional RNNs) into ResNet architecture, in order to better extract spatio-temporal information and improve the performance on image classification tasks. The model is built with PyTorch library.

### Introduction

ResNet is one of the most outstanding architectures in the field of image classification. In the past, neural networks with deep layers do not guarantee better results. This is because the propagation between layers comes with inevitable loss of information and thus making identical mapping between layers impossible. ResNet solves this issue by manually passing through the input and adding up with the output via 'shortcuts' and this forces the model to learn residuals instead. With ResNet, deep networks has a great chance of outperforming shallow networks because more layer enables more complex features being extracted and learned.

'However, the shortcut connection mechanism makes each block focus on learning its respective residual output, where the inner block information communication is somehow ignored and some reusable information learned from previous blocks tends to be forgotten in later blocks.'(Xu et al., 2022).

The authors of RegNet argues that ResNet is not effectively learning because the 'shortcut' connection leads to very similar learned features between adjacent layers. Their solution is to have regulator modules in parallel with the ResNet design. The chosen design of the regulators are convolutional RNNs,  convolutional RNNs use a memory mechanism to store and pass on the complementary spatio-temporal information from one layer to the other.
In their paper, the result shows that RegNet is able to achieve the same performance as ResNet in a less computationally expensive manner. In this report, we are evaluating RegNet's performance on the STL10 dataset.

### Data

The STL10 dataset contains 5000 training data and 8000 test data. We further divide the training data into 80% training and 20% validation.

The image size is (96,96) with RGB colour channels.

Normalization is applied. Normalization is beneficial because it ease the training process and effectively reduce the chance of overfitting.

10 classes: 0:'airplane', 1:'bird', 2:'car', 3:'cat', 4:'deer', 5:'dog', 6:'horse', 7:'monkey', 8:'ship', 9:'truck'

### Method

To improve the model performance, the authors point to the SE-ResNet architecture in their paper[2]. SE stands for Squeeze and Excitation layer, 'it adaptively re-calibrates channel-wise feature responses by explicitly modelling interdependencies between channels.'(Hu et al., 2018). SE layer is composed of squeeze step and excitation step. In squeeze step, each channel of the input data is mapped to a single value by a global average pooling layer. The value is used to compute the weights in the excitation step. In excitation step, original channels are scaled by the weights, where higher weights imply more importance given. In this way, SE layer emphasizes important features from learned feature maps. The position of SE layers are the same as they are in the SE-ResNet, specifically after the learned feature maps and before the addition to the input.

The convolutional RNN model implemented in our report is convolutional LSTM. LSTM retrieves and stores information (hidden states H and C) over time using a gating system. The only difference between convolutional LSTM and regular LSTM is that the linear layers are replaced by convolutional layers.

The visualization of a single RegNet block can be found in the repo. In our design, we choose to have 3 of these blocks followed by a fully connected layer.

### Reference

1. Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer Networks in Unsupervised Feature Learning AISTATS, 2011.

2. Xu, J., Pan, Y., Pan, X., Hoi, S., Yi, Z., &amp; Xu, Z. (2022). RegNet: Self-regulated network for Image Classification. IEEE Transactions on Neural Networks and Learning Systems, 1–6. https://doi.org/10.1109/tnnls.2022.3158966 

3. Hu, J., Shen, L., &amp; Sun, G. (2018). Squeeze-and-excitation networks. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2018.00745 

4. Squeeze and excitation networks explained with pytorch implementation. Committed towards better future. (2020, July 24). Retrieved December 16, 2022, from https://amaarora.github.io/2020/07/24/SeNet.html 

5. Esposito, P. (2020, May 25). Building a LSTM by hand on pytorch. Medium. Retrieved December 16, 2022, from https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091 
