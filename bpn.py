import numpy as np
import matplotlib.pyplot as plt
from data_preparation import load_titanic, load_diabetes


# ============================================================
#  BACKPROPAGATION NEURAL NETWORK (BPN) - From Scratch
# ============================================================
class BPN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the BPN with random weights and biases.
        
        Parameters:
            input_size    : number of input features
            hidden_size   : number of neurons in the hidden layer
            output_size   : number of output classes
            learning_rate : step size for weight updates
        """
        self.lr = learning_rate

        # Weights initialized with small random values (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Training history
        self.loss_history = []
        self.acc_history  = []

    # ── Activation functions ─────────────────────────────────
    def sigmoid(self, z):
        """Sigmoid activation: squashes values between 0 and 1."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        """Derivative of sigmoid (used in backpropagation)."""
        return a * (1 - a)

    # ── Forward propagation ──────────────────────────────────
    def forward(self, X):
        """
        Pass input data through the network to get predictions.
        
        Input Layer → Hidden Layer → Output Layer
        """
        # Input → Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Hidden → Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    # ── Backward propagation ─────────────────────────────────
    def backward(self, X, y_true):
        """
        Propagate error backward through the network and update weights.
        """
        n = X.shape[0]

        # Convert labels to column vector
        y = y_true.reshape(-1, 1)

        # Output layer error
        delta2 = (self.a2 - y) * self.sigmoid_derivative(self.a2)

        # Hidden layer error
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 -= self.lr * np.dot(self.a1.T, delta2) / n
        self.b2 -= self.lr * np.sum(delta2, axis=0, keepdims=True) / n
        self.W1 -= self.lr * np.dot(X.T, delta1) / n
        self.b1 -= self.lr * np.sum(delta1, axis=0, keepdims=True) / n

    # ── Loss function ────────────────────────────────────────
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        y      = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # ── Training ─────────────────────────────────────────────
    def train(self, X_train, y_train, epochs=1000, min_error=None, verbose=True):
        """
        Train the network.

        Stop conditions:
            epochs    : maximum number of iterations
            min_error : stop early if loss drops below this threshold
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train)

            # Compute loss and accuracy
            loss = self.compute_loss(y_train, y_pred)
            acc  = self.accuracy(y_train, y_pred)

            self.loss_history.append(loss)
            self.acc_history.append(acc)

            # Backward pass
            self.backward(X_train, y_train)

            # Print progress every 100 epochs
            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

            # Stop condition: minimum error reached
            if min_error is not None and loss < min_error:
                print(f"  ✅ Stop: loss reached {loss:.4f} at epoch {epoch}")
                break

    # ── Prediction ───────────────────────────────────────────
    def predict(self, X):
        """Predict class labels for input X."""
        output = self.forward(X)
        return (output >= 0.5).astype(int).flatten()

    # ── Accuracy ─────────────────────────────────────────────
    def accuracy(self, y_true, y_pred):
        """Calculate prediction accuracy."""
        predictions = (y_pred >= 0.5).astype(int).flatten()
        return np.mean(predictions == y_true)


# ============================================================
#  EXPERIMENT: Test different hidden sizes and learning rates
# ============================================================
def run_experiment(X_train, X_test, y_train, y_test, dataset_name):
    """
    Run BPN with different hidden sizes and learning rates.
    Print a summary table with results.
    """
    hidden_sizes    = [5, 10, 20]
    learning_rates  = [0.01, 0.05, 0.1]
    epochs          = 1000
    min_error       = 0.01

    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")
    print(f"  {'Hidden':>8} | {'LR':>6} | {'Train Acc':>10} | {'Test Acc':>10}")
    print(f"  {'-'*46}")

    best_acc   = 0
    best_model = None
    best_cfg   = {}

    results = []

    for hs in hidden_sizes:
        for lr in learning_rates:
            # Set random seed for reproducibility
            np.random.seed(42)

            model = BPN(
                input_size  = X_train.shape[1],
                hidden_size = hs,
                output_size = 1,
                learning_rate = lr
            )

            model.train(X_train, y_train, epochs=epochs, min_error=min_error, verbose=False)

            train_acc = model.accuracy(y_train, model.forward(X_train))
            test_acc  = model.accuracy(y_test,  model.forward(X_test))

            print(f"  {hs:>8} | {lr:>6} | {train_acc*100:>9.2f}% | {test_acc*100:>9.2f}%")

            results.append((hs, lr, train_acc, test_acc, model))

            if test_acc > best_acc:
                best_acc   = test_acc
                best_model = model
                best_cfg   = {'hidden_size': hs, 'learning_rate': lr}

    print(f"\n  ⭐ Best config → Hidden: {best_cfg['hidden_size']} | LR: {best_cfg['learning_rate']} | Test Acc: {best_acc*100:.2f}%")

    return best_model, results


# ============================================================
#  PLOT: Loss and accuracy curves
# ============================================================
def plot_training(model, dataset_name):
    """Plot loss and accuracy curves for the best model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(model.loss_history, color='tomato')
    ax1.set_title(f'Loss curve — {dataset_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(model.acc_history, color='steelblue')
    ax2.set_title(f'Accuracy curve — {dataset_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'bpn_{dataset_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
    print(f"  📊 Plot saved as bpn_{dataset_name.lower().replace(' ', '_')}.png")


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  FASE 2 — Backpropagation Neural Network (BPN)")
    print("=" * 60)

    # Load datasets
    X_train_t, X_test_t, y_train_t, y_test_t = load_titanic('datasets/Titanic-Dataset.csv')
    X_train_d, X_test_d, y_train_d, y_test_d = load_diabetes('datasets/diabetes.csv')

    # Run experiments
    best_titanic,  results_t = run_experiment(X_train_t, X_test_t, y_train_t, y_test_t, "Titanic")
    best_diabetes, results_d = run_experiment(X_train_d, X_test_d, y_train_d, y_test_d, "Diabetes")

    # Plot best models
    plot_training(best_titanic,  "Titanic")
    plot_training(best_diabetes, "Diabetes")