# MNIST Neural Network From Scratch

A two-layer neural network implementation built from scratch using only NumPy, designed to classify handwritten digits from the MNIST dataset. This project serves as an educational tool to understand the mathematical foundations of neural networks.

## 🎥 Video Tutorial

Watch the complete implementation and mathematical explanation: [YouTube Video](https://youtu.be/w8yWXqWQYmU)

## 🎯 Results

- **Training Accuracy:** 85.2%
- **Validation Accuracy:** 84.4%
- **Training Time:** ~500 iterations
- **Model Size:** Lightweight (10 hidden units)

## 🏗️ Architecture

```
Input Layer (784 units) → Hidden Layer (10 units, ReLU) → Output Layer (10 units, Softmax)
```

### Network Specifications:
- **Input:** 784 features (28×28 pixel images, flattened)
- **Hidden Layer:** 10 units with ReLU activation
- **Output Layer:** 10 units with Softmax activation (digit classes 0-9)
- **Optimizer:** Gradient Descent
- **Learning Rate:** 0.1

## 🧮 Mathematical Foundation

### Forward Propagation
```
Z[1] = W[1]X + b[1]
A[1] = ReLU(Z[1])
Z[2] = W[2]A[1] + b[2]
A[2] = Softmax(Z[2])
```

### Backward Propagation
```
dZ[2] = A[2] - Y
dW[2] = (1/m) * dZ[2] * A[1]^T
dZ[1] = W[2]^T * dZ[2] * ReLU'(Z[1])
dW[1] = (1/m) * dZ[1] * X^T
```

### Parameter Updates
```
W[1] := W[1] - α * dW[1]
W[2] := W[2] - α * dW[2]
b[1] := b[1] - α * db[1]
b[2] := b[2] - α * db[2]
```

## 📦 Installation

### Prerequisites
- Python 3.7+
- NumPy
- Pandas
- Matplotlib

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-neural-network-from-scratch.git
cd mnist-neural-network-from-scratch

# Install dependencies
pip install -r requirements.txt

# Download MNIST dataset
# Place train.csv in the data/ folder (available from Kaggle MNIST competition)
```

## 🚀 Quick Start

```python
from src.neural_network import TwoLayerNN
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data/train.csv')
data = np.array(data)

# Prepare training data
X_train, Y_train = prepare_data(data[1000:])
X_dev, Y_dev = prepare_data(data[:1000])

# Initialize and train model
nn = TwoLayerNN(input_size=784, hidden_size=10, output_size=10)
nn.train(X_train, Y_train, epochs=500, learning_rate=0.1)

# Make predictions
accuracy = nn.evaluate(X_dev, Y_dev)
print(f"Validation Accuracy: {accuracy:.3f}")
```

## 📁 Project Structure

```
mnist-neural-network/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   └── neural_network.py      # Main neural network implementation
│
├── data/
│   └── README.md              # Data download instructions
│
├── examples/
│   ├── train_model.py         # Training example
│   └── visualize_results.py   # Visualization utilities
│
└── docs/
    └── math_derivations.md    # Detailed mathematical explanations
```

## 🔧 Key Features

- **From Scratch Implementation:** No external ML libraries (TensorFlow, PyTorch, etc.)
- **Educational Focus:** Clear, readable code with extensive comments
- **Mathematical Transparency:** All operations implemented explicitly
- **Visualization Tools:** Plot training progress and sample predictions
- **Modular Design:** Easy to modify and extend

## 📊 Performance Metrics

| Metric | Training Set | Validation Set |
|--------|--------------|----------------|
| Accuracy | 85.2% | 84.4% |
| Loss | 0.45 | 0.48 |
| Training Time | ~30 seconds | - |

## 🎨 Sample Predictions

The model successfully predicts handwritten digits with 84% accuracy on unseen data. Here are some example predictions:

- Image 1: Predicted = 3, Actual = 2 ❌
- Image 2: Predicted = 1, Actual = 1 ✅
- Image 3: Predicted = 4, Actual = 4 ✅
- Image 4: Predicted = 2, Actual = 2 ✅

## 🔍 Understanding the Code

### Core Components

1. **Activation Functions**
   - ReLU for hidden layer
   - Softmax for output layer

2. **Loss Function**
   - Cross-entropy loss (implicit in softmax derivative)

3. **Optimization**
   - Mini-batch gradient descent
   - Fixed learning rate

4. **Data Processing**
   - Normalization (pixel values / 255)
   - One-hot encoding for labels

## 🤝 Contributing

Contributions are welcome! Here are some ways to improve the project:

- Add different activation functions (tanh, sigmoid, etc.)
- Implement different optimizers (Adam, RMSprop)
- Add regularization techniques (dropout, L2 regularization)
- Create more visualization tools
- Improve documentation

### Development Setup
```bash
# Fork the repository and clone your fork
git clone https://github.com/yourusername/mnist-neural-network-from-scratch.git
cd mnist-neural-network-from-scratch

# Create a new branch for your feature
git checkout -b feature-name

# Make your changes and commit
git commit -am "Add new feature"

# Push to your fork and create a pull request
git push origin feature-name
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset creators and Kaggle for hosting
- The machine learning community for educational resources
- Viewers and contributors who help improve this project

---

⭐ **Star this repository if you found it helpful for learning neural networks!**
