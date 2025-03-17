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

1. Ensure Python is installed and install the required dependencies:  

```bash
pip install numpy matplotlib seaborn scikit-learn wandb keras
```
2. Ensure requirements includes:
```bash
numpy
matplotlib
seaborn
scikit-learn
tensorflow
wandb
```
3. Set up Wandb:
Sign up for a free account at wandb.ai.
```bash
wandb login
```
## ** Notebook `DL_Assignment_1.ipynb` Specification **
 - Downloading the Fashion-MNIST
 - Normalize the data
 - Ploting image class
 - Feedforward Neural network to generate the probability distribution
 - Optimization Functions
 - Implimentation of Backpropagation
 - Plotting the accuracy using wandb
 - Plotting the confusion matrix in wandb For the best model
 - ploting the accuracy for the MSE loss function
 - Downloading the MNIST-data
 - Normalizing the data
 - Testing accuracy for the data with three best model

## Evaluate the Model
- After training, the model evaluates on the test dataset by:
   - Computing test accuracy
   - Generating a confusion matrix
   - Logging results in wandb
- Viewing Results
  - Check Accuracy & Loss: Logged to wandb
  - View Confusion Matrix: Plotted using Seaborn
  - Compare Hyperparameter Runs: On the wandb dashboard
 
## Supported Argument 

| Argument | Default Value | Description |
|----------|--------------|-------------|
| `-wp`, `--wandb_project` | `myprojectname` | Project name for Wandb dashboard |
| `-we`, `--wandb_entity` | `myname` | Wandb entity (username or team) |
| `-d`, `--dataset` | `fashion_mnist` | Dataset to use: `mnist` or `fashion_mnist` |
| `-e`, `--epochs` | `10` | Number of epochs to train |
| `-b`, `--batch_size` | `16` | Batch size for training |
| `-l`, `--loss` | `cross_entropy` | Loss function: `mean_squared_error` or `cross_entropy` |
| `-o`, `--optimizer` | `adam` | Optimizer: `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `-lr`, `--learning_rate` | `0.0001` | Learning rate for the optimizer |
| `-m`, `--momentum` | `0.9` | Momentum for `momentum` and `NAG` optimizers |
| `-beta`, `--beta` | `0.9` | Beta for `RMSProp` optimizer |
| `-beta1`, `--beta1` | `0.9` | Beta1 for `Adam` and `Nadam` optimizers |
| `-beta2`, `--beta2` | `0.999` | Beta2 for `Adam` and `Nadam` optimizers |
| `-eps`, `--epsilon` | `1e-8` | Epsilon for numerical stability in optimizers |
| `-w_d`, `--weight_decay` | `0.0` | Weight decay (L2 regularization) coefficient |
| `-w_i`, `--weight_init` | `xavier` | Weight initialization: `random` or `xavier` |
| `-nhl`, `--num_layers` | `4` | Number of hidden layers |
| `-sz`, `--hidden_size` | `128` | Number of neurons per hidden layer |
| `-a`, `--activation` | `relu` | Activation function: `identity`, `sigmoid`, `tanh`, `relu` |
## Self Declaration
I, Anvit Kumar, swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.

Wandb Link  : https://api.wandb.ai/links/ma24m004-iit-madras/g89ewehu
GitHub link : https://github.com/Anvit131/Da6401_assignment1
