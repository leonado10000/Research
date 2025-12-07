class CSLoss:
    def __init__(self, model):
        self.model = model
        self.epsilon = 1e-12  # small constant to avoid log(0)

    def forward(self, y_pred, y_true):
        """
        Compute the cross-entropy loss.

        Parameters:
        y_pred (numpy.ndarray): Predicted probabilities (output of softmax), shape (batch_size, num_classes)
        y_true (numpy.ndarray): True labels in one-hot encoded format, shape (batch_size, num_classes)

        Returns:
        float: Cross-entropy loss
        """