# Number Predictor

<div align="center">
  <img src="https://sammyurfen.github.io/Number_predictor/icon.webp" alt="Number Predictor Logo" width="200"/>
  <br>
  <a href="https://sammyurfen.github.io/Number_predictor/">Live Demo</a>
</div>

## Overview
Number Predictor is an interactive web application that uses machine learning to recognize hand-drawn digits. Draw any number (0-9) on the 28x28 grid, and the application will predict which digit you've drawn using a pre-trained MNIST neural network model.

**Note:** The model may not always provide accurate predictions, especially for ambiguous or unclear drawings. Please use it as a demonstration of the technology rather than a definitive tool.

## Features
- 🎨 **Interactive 28x28 drawing grid**
- 🤖 **Real-time digit prediction using MNIST model**
- ✨ **Animated scanning effect during prediction**
- 🔄 **One-click grid clearing**
- 📱 **Responsive design**

## Tech Stack
- **HTML5 Canvas**
- **ONNX Runtime Web**
- **PyTorch**
- **Vanilla JavaScript**
- **CSS3**

## Installation

```bash
git clone https://github.com/sammyurfen/Number_predictor.git
cd Number_predictor
```

## Usage
1. Visit the [Live Demo](https://sammyurfen.github.io/Number_predictor/).
2. Draw a number (0-9) on the grid.
3. Click **"Predict Number"**.
4. Use **"Clear"** to reset the grid.

## Project Structure

```
.
├── index.html              # Main HTML file
├── style.css               # Styling
├── script.js               # Frontend logic
├── mnist_model.onnx        # Exported ONNX model
├── mnist_model.pth         # PyTorch model
└── mnist_trainer.py        # Model training script
```

## Downloading the MNIST Dataset
To train the model, you need to download the MNIST dataset. You can download it from the [official MNIST website](http://yann.lecun.com/exdb/mnist/).

Download the following files:

- [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
- [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
- [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
- [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

Extract the files and place them in an `data` directory within your project.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
**Sammy Urfen**  
GitHub: [SammyUrfen](https://github.com/sammyurfen)  

Project Link: [https://github.com/sammyurfen/Number_predictor](https://github.com/sammyurfen/Number_predictor)
