# **Deep Learning Fashion-MNIST Classifier**  

This project implements a **fully connected neural network** from scratch using **NumPy** to classify images from the **Fashion-MNIST** dataset. The model supports different activation functions, weight initialization techniques, and optimizers. **Weights & Biases (wandb)** is used for **experiment tracking** and **hyperparameter tuning**.  

## ** Features**  

- **Dataset**: Fashion-MNIST (10-class grayscale 28x28 images)  
- **Preprocessing**:
- Normalization
- Reshaping  
- **Custom Neural Network**:  
  - Supports **ReLU, Sigmoid, and Tanh** activation functions  
  - Weight initialization options (**Xavier, Random**)  
- **Optimizers Implemented**:
  - **SGD, Momentum, Nesterov, RMSProp, Adam, Nadam**  
- **Loss Functions**:
  - **Cross-entropy loss** (default)  
  - **Mean Squared Error (MSE)** (alternative)  
- **Hyperparameter Optimization**:
  - **Bayesian sweep** using **wandb**  
  - Tunable parameters:  
    - Learning rate, batch size, weight decay, optimizer, activation function, number of hidden layers, etc.  
- **Evaluation**:
  - Computes **training & validation accuracy** 
  - **Confusion matrix visualization** with Seaborn  and 
  - **Results logged** to **wandb** to understand experimentally for better understanding

---

## ** Installation**  

Ensure Python is installed and install the required dependencies:  

```bash
pip install numpy matplotlib seaborn scikit-learn wandb keras

## ** Train the Model **
To train the neural network on Fashion-MNIST, run:
python DL_1.ipynb


