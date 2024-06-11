import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=131, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        # Apply first convolutional layer with ReLU activation and max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        # Apply second convolutional layer with ReLU activation and max pooling
        x = self.pool2(F.relu(self.conv2(x)))
        # Apply third convolutional layer with ReLU activation and max pooling
        x = self.pool3(F.relu(self.conv3(x)))
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply output layer with sigmoid activation
        x = torch.sigmoid(self.fc4(x))
        return x

# Example usage
if __name__ == "__main__":
    # Create an instance of the CNN2D model
    model = CNN2D()
    #model2 = Resnet(in_channels=131, num_class=2)
    # Print the model architecture
    print(model)

    # Example input tensor (batch_size=1, channels=131, height=17, width=17)
    input_tensor = torch.randn(100, 131, 17, 17)
    # Get the model output
    output = model(input_tensor)
    #output2 = model2(input_tensor)
    # Print the output
    print(output.shape)
    #print(output2.shape)
