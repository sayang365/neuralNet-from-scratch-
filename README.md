# MNIST neuralNet from Scratch

A two-layer neural network built from scratch using only NumPy for MNIST digit classification. Achieves **84% accuracy** on validation data.

## Architecture

```
Input (784) → Hidden (10, ReLU) → Output (10, Softmax)
```

## Mathematics

**Forward Propagation:**
```
Z¹ = W¹X + b¹
A¹ = ReLU(Z¹)
Z² = W²A¹ + b²
A² = Softmax(Z²)
```

**Backward Propagation:**
```
dZ² = A² - Y
dW² = (1/m) × dZ² × A¹ᵀ
db² = (1/m) × Σ(dZ²)
dZ¹ = W²ᵀ × dZ² ⊙ ReLU'(Z¹)
dW¹ = (1/m) × dZ¹ × Xᵀ
db¹ = (1/m) × Σ(dZ¹)
```

**Parameter Updates:**
```
W¹ := W¹ - α × dW¹
W² := W² - α × dW²
b¹ := b¹ - α × db¹
b² := b² - α × db²
```

## Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Download MNIST train.csv from Kaggle to data/ folder
# Run training
python examples/train_model.py
```

## Usage

```python
from src.neural_network import TwoLayerNN, load_data, prepare_data

# Load and prepare data
data = load_data('data/train.csv')
X_train, Y_train = prepare_data(data[1000:])
X_dev, Y_dev = prepare_data(data[:1000])

# Train model
nn = TwoLayerNN(784, 10, 10)
nn.train(X_train, Y_train, epochs=500, learning_rate=0.1)

# Evaluate
accuracy = nn.evaluate(X_dev, Y_dev)
print(f"Accuracy: {accuracy:.3f}")
```

## Results

- **Training Accuracy:** 85.2%
- **Validation Accuracy:** 84.4%
- **Parameters:** ~8K (10×784 + 10×10 + biases)
- **Training Time:** ~30 seconds

## Requirements

- Python 3.7+
- NumPy ≥ 1.21.0
- Pandas ≥ 1.3.0  
- Matplotlib ≥ 3.5.0

