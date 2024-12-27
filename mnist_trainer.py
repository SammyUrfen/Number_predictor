import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Define the model architecture
class NoiseDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = self.generate_scribbles(num_samples)
        self.targets = torch.full((num_samples,), 10, dtype=torch.int64)

    def generate_scribbles(self, num_samples):
        scribbles = []
        for _ in range(num_samples):
            image = np.zeros((28, 28), dtype=np.float32)
            num_lines = np.random.randint(5, 15)
            for _ in range(num_lines):
                x0, y0 = np.random.randint(0, 28, size=2)
                x1, y1 = np.random.randint(0, 28, size=2)
                rr, cc = self.bresenham_line(x0, y0, x1, y1)
                image[rr, cc] = 1.0
            scribbles.append(image)
        return torch.tensor(np.array(scribbles)).unsqueeze(1)

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's Line Algorithm to generate line coordinates."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        rr, cc = [], []
        while True:
            rr.append(y0)
            cc.append(x0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return rr, cc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        return image, int(label)  # Ensure label is an integer

def custom_normalize(tensor):
    tensor = tensor.numpy().flatten()  # Convert to NumPy array and flatten
    normalized_tensor = np.zeros_like(tensor)
    
    normalized_tensor[tensor < 0.2] = -1
    normalized_tensor[(tensor >= 0.2) & (tensor < 0.5)] = -0.41421356237309515
    normalized_tensor[(tensor >= 0.5) & (tensor < 0.8)] = 0
    normalized_tensor[tensor >= 0.8] = 1
    
    return torch.tensor(normalized_tensor).view(1, 28, 28)  # Convert back to tensor and reshape

def normalize_dataset(dataset):
    for i in range(len(dataset)):
        image, label = dataset[i]
        dataset.data[i] = custom_normalize(image)
    return dataset

def get_trainset():
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    # trainset = normalize_dataset(trainset)
    # noise_trainset = NoiseDataset(num_samples=6000)
    # return DataLoader(ConcatDataset([trainset, noise_trainset]), batch_size=64, shuffle=True)
    return DataLoader(trainset, batch_size=64, shuffle=True)

def get_testset():
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    # testset = normalize_dataset(testset)
    # noise_testset = NoiseDataset(num_samples=1000)
    # return DataLoader(ConcatDataset([testset, noise_testset]), batch_size=64, shuffle=False)
    return DataLoader(testset, batch_size=64, shuffle=True)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 0-9 digits + 10 for blank digit
        )

    def forward(self, x):
        return self.fc(x)

# Main function
def main():
    """
    Main function to train and evaluate an MNIST model using PyTorch.
    
    - Defines a data transformation pipeline for normalization.
    - Loads training and test datasets, including noise datasets.
    - Initializes the MNIST model, loss function, and optimizer.
    - Trains the model for a specified number of epochs, printing the loss.
    - Saves the trained model's state to a file.
    - Exports the trained model to the ONNX format.
    - Evaluates the model's accuracy on the test dataset and prints the result.
    """

    # Create separate transforms for MNIST and Noise datasets
    trainloader = get_trainset()
    testloader = get_testset()

    # Initialize model, loss, optimizer
    model = MNISTModel()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            # Reshape images to (batch_size, 784)
            images = images.view(images.shape[0], -1)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}')

    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')

    # Export to ONNX
    dummy_input = torch.randn(1, 784)
    torch.onnx.export(model, dummy_input, "mnist_model.onnx",
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})
    
    # Test model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total}%')


if __name__ == "__main__":
    main()