from core.tensor_v1 import Tensor

class MSE:
    """
    model needs to have input_dim and output_dim attributes
    """
    def __init__(self):
        self.name = "MeanSquareError"
        self.n = 1
        self.X = None
        self.result = []
        self.total_loss = 0

    def sum_all(self, x):
        """Flatten and sum all numbers from nested mse structure."""
        if isinstance(x, (int, float)):
            return x
        return sum(self.sum_all(i) for i in x)

    def __call__(self, prediction:Tensor, target:Tensor):
        """
        recursive mean square error calculation
        returns:
            - mse tensor (same structure as prediction)
            - scalar total mse
        """ 
        devisor = 1
        for x in prediction.shape:
            devisor *= x

        err = prediction - target
        # loss tensor 1 -> gets the mse error value; to get the loss value
        # loss tensor 2 -> gets backprop value
        sq = err*err

        total_loss = self.sum_all(sq.data)
        err = err*(1/devisor)
        err.origin =  self
        err.item = lambda : total_loss

        return err

# a = [[[[1,2],[3,4]],
#      [[5,6],[7,8]]], 
#     [[[1,1],[1,1]],
#      [[1,1],[1,1]]]]

# b = [[[[0,0],[0,0]],
#      [[0,0],[0,0]]],
#     [[[0,0],[0,0]],
#      [[0,0],[0,0]]]]

# mse = MSE()
# print("MSE ====> ", mse.forward(a, b))