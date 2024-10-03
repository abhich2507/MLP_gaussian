import numpy as np




##cross entropy loss
def ce_loss(probs, labels):
    """
    Compute the cross-entropy loss.
    probs: numpy array of shape (N, C) where N is the number of samples and C is the number of classes.
    This is the output of the softmax function, i.e., predicted probabilities.
    labels: numpy array of shape (N, C), one-hot encoded true labels.
    Returns:
    loss: scalar, the average cross-entropy loss over all samples.
    """
    # Clip the probabilities to prevent log(0) which can result in NaN values
    probs = np.clip(probs, 1e-12, 1.0)
    # Compute the cross-entropy loss
    loss = -np.sum(labels * np.log(probs)) / labels.shape[0]
    return loss

##derivative of cross entropy loss
def derivative_ce(probs, labels):
    """
    Compute the derivative of the cross-entropy loss w.r.t. logits (softmax input).

    probs: numpy array of shape (N, C), the predicted probabilities (output of softmax).
    labels: numpy array of shape (N, C), one-hot encoded true labels.

    Returns:
    dL/dz: numpy array of shape (N, C), the derivative of the loss w.r.t the logits.
    """
    return (probs - labels) / labels.shape[0]



##derivative of softmax
def derivative_softmax(probs, labels):
    """
    Compute the derivative of the cross-entropy loss w.r.t. logits (softmax input).
    probs: numpy array of shape (N, C), the predicted probabilities (output of softmax).
    labels: numpy array of shape (N, C), one-hot encoded true labels.
    Returns:
    dL/dz: numpy array of shape (N, C), the derivative of the loss w.r.t the logits.
    """
    return (probs - labels) / labels.shape[0]

##relu activation function



##derivative of relu



##one hot encoding




## Define the Neural Network class

import numpy as np

class InitializeModel:
    def __init__(self, random_seed=123):
        print("Model initialization")
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

class MLP(InitializeModel):
    
    def __init__(self, num_features, hidden_layers, num_classes=1, random_seed=123):
        super().__init__(random_seed)
        
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_layers = hidden_layers
        
        # Initialize weights and biases
        self.initialize_weights()
       

    # def initialize_weights(self):
    #     # Initialize weights and biases for hidden layers
    #     self.weights = []
    #     self.biases = []
        
    #     # First hidden layer
    #     self.weights.append(self.rng.normal(0, 0.1, (self.hidden_layers[0], self.num_features)))
    #     self.biases.append(np.zeros((self.hidden_layers[0])))
        
    #     # Hidden layers
    #     for i in range(1, len(self.hidden_layers)):
    #         self.weights.append(self.rng.normal(0, 0.1, (self.hidden_layers[i], self.hidden_layers[i - 1])))
    #         self.biases.append(np.zeros((self.hidden_layers[i])))
        
    #     # Output layer
    #     self.weights.append(self.rng.normal(0, 0.1, (self.num_classes, self.hidden_layers[-1])))
    #     self.biases.append(np.zeros((self.num_classes)))

    #     # Store metrics for bookkeeping
    #     self.metrics = dict()
        
    #     print("Model initialized")
    #     self._print_weights_and_biases()

    def initialize_weights(self):
        """
        Initializes weights and biases for the network using He initialization for ReLU-based activations.
        For the output layer, we use Xavier initialization.
        """
        self.weights = []
        self.biases = []
        
        # He Initialization for hidden layers (best for ReLU)
        def he_init(size_in, size_out):
            return self.rng.normal(0, np.sqrt(2.0 / size_in), (size_out, size_in))
        
        # Xavier Initialization for output layer (works well for softmax/tanh)
        def xavier_init(size_in, size_out):
            return self.rng.normal(0, np.sqrt(1.0 / size_in), (size_out, size_in))

        # First hidden layer
        self.weights.append(he_init(self.num_features, self.hidden_layers[0]))
        self.biases.append(np.zeros((self.hidden_layers[0])))

        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.weights.append(he_init(self.hidden_layers[i - 1], self.hidden_layers[i]))
            self.biases.append(np.zeros((self.hidden_layers[i])))

        # Output layer
        self.weights.append(xavier_init(self.hidden_layers[-1], self.num_classes))
        self.biases.append(np.zeros((self.num_classes)))

        # Store metrics for bookkeeping
        self.metrics = dict()

        print("Model initialized with He and Xavier initialization.")
        # self._print_weights_and_biases()



    # def _print_weights_and_biases(self):
    #     # for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
    #         # print(f"Layer {i + 1} weights:\n", weight)
    #         # print(f"Shape of layer {i + 1} weights:", weight.shape)
    #         # print(f"Bias for layer {i + 1}:", bias.shape)

    def reinitialize(self, random_seed=None):
        """Reinitialize the model weights and biases if you need to."""
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        self.initialize_weights()
    
    def softmax(self,logits):
        """
        Compute softmax for each set of logits with numerical stability.
        Parameters:
        logits (ndarray): Raw logits of shape (N, C) where N is the number of samples and C is the number of classes.
        Returns:
        ndarray: Softmax probabilities of shape (N, C).
        """
        # Check for NaN or inf values in logits (optional for debugging)
        # print("Logits:", logits)
        if np.isnan(logits).any() or np.isinf(logits).any():
            print("Warning: Logits contain NaN or infinity values!")
        # Clip the logits to avoid extremely large or small values that might cause overflows in exp
        logits = np.clip(logits, -100, 100)
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def derivative_softmax(self,probs, labels):
        """
        Compute the derivative of the cross-entropy loss w.r.t. logits (softmax input).
        probs: numpy array of shape (N, C), the predicted probabilities (output of softmax).
        labels: numpy array of shape (N, C), one-hot encoded true labels.
        Returns:
        dL/dz: numpy array of shape (N, C), the derivative of the loss w.r.t the logits.
        """
        return (probs - labels) / labels.shape[0]
    def relu(self,x):
        return np.maximum(0, x)
    
    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)

    def int_to_onehot(self,y, num_classes):
        onehot = np.zeros((len(y), num_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot

    def forward(self, x):
        # print("X:", x)
        # print("X: shape:", x.shape)
        x=(x-np.min(x))/(np.max(x)-np.min(x))
        self.activations = [x]
        
        # Forward through all hidden layers
        for i in range(len(self.hidden_layers)):
            # print("activations:", self.activations[-1], "weights:", self.weights[i].T, "biases:", self.biases[i])
            z = np.dot(self.activations[-1], self.weights[i].T) + self.biases[i]  # Shape: (batch_size, num_units)
            # print("Z:", z)
            # print("Z: shape:", z.shape)
            a = self.relu(z)  # Activation function
            self.activations.append(a)
        
        # Output Layer
        z_out = np.dot(self.activations[-1], self.weights[-1].T) + self.biases[-1]  # Shape: (batch_size, num_classes)
        # print("Z_out:", z_out)
        a_out = self.softmax(z_out)  # Output activation function
        return a_out

    # def compute_loss(self, y_pred, y_true):
    #     """Computes Mean Squared Error loss."""
    #     loss = np.mean((y_pred - y_true) ** 2)
    #     return loss
    
    def compute_loss(self,probs, labels):
        """
        Compute the cross-entropy loss.
        probs: numpy array of shape (N, C) where N is the number of samples and C is the number of classes.
        This is the output of the softmax function, i.e., predicted probabilities.
        labels: numpy array of shape (N, C), one-hot encoded true labels.
        Returns:
        loss: scalar, the average cross-entropy loss over all samples.
        """
        # Clip the probabilities to prevent log(0) which can result in NaN values
        probs = np.clip(probs, 1e-12, 1.0)
        # Compute the cross-entropy loss
        loss = -np.sum(labels * np.log(probs)) / labels.shape[0]
        return loss

    def backward(self, x, y, learning_rate=0.0005,clip_value=1.0):
    # Forward pass to compute the output
        y_onehot = self.int_to_onehot(y, self.num_classes)  # Convert labels to one-hot
        output = self.forward(x)

        # Compute the error
        error = output - y_onehot  # Shape: (batch_size, num_classes)
        
        loss = self.compute_loss(output, y_onehot)
        self.metrics['loss'] = self.metrics.get('loss', []) + [loss]

        # Backpropagation
        delta = self.derivative_softmax(output, y_onehot)  # Use the adjusted function

        # Update weights and biases from output layer to input
        for i in reversed(range(len(self.hidden_layers) + 1)):
            if i == len(self.hidden_layers):  # Output layer
                d_a_out = delta
            else:  # Hidden layers
                d_a = np.dot(delta, self.weights[i + 1])  # Shape: (batch_size, num_units in next layer)
                d_z = self.relu_derivative(self.activations[i + 1])  # Shape: (batch_size, num_units)
                delta = d_a * d_z  # Shape: (batch_size, num_units)

            # Update weights and biases
            if i == 0:
                d_loss_weights = np.dot(delta.T, x)  # Shape: (num_units, num_features)
            else:
                d_loss_weights = np.dot(delta.T, self.activations[i])  # Shape: (num_units, num_units_previous)

            d_loss_bias = np.sum(delta, axis=0)  # Shape: (num_units,)

            d_loss_weights = np.clip(d_loss_weights, -clip_value, clip_value)
            d_loss_bias = np.clip(d_loss_bias, -clip_value, clip_value)
            
            self.weights[i] -= learning_rate * d_loss_weights
            self.biases[i] -= learning_rate * d_loss_bias

    




