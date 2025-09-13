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

