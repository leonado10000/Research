import numpy as np

class Pool:
    def __init__(self, kernel = 2, stride = 2, mode="max"):
        self.kernel = kernel
        self.stride = stride
        self.mode = mode.lower()

    def _window_pool(self, w):
        if self.mode == "max":
            return max([max(row) for row in w])
        elif self.mode == "avg":
            return sum([sum(row) for row in w])/ (len(w) * len(w[0]))

    def forward(self, X):
        X = np.array(X)
        batch_size, in_c, h, w = X.shape

        h_out = (h - self.kernel) // self.stride + 1
        w_out = (w - self.kernel) // self.stride + 1
        
        out = np.zeros((batch_size, in_c, h_out, w_out))

        for b in range(batch_size):
            for c in range(in_c):
                for i in range(h_out):
                    for j in range(w_out):
                        window = [
                            [
                                X[b][c][i*self.stride + m][j*self.stride + n] 
                                for n in range(self.kernel) 
                                for m in range(self.kernel)
                             ]
                        ]
                        out[b][c][i][j] = self._window_pool(window)
        return np.array(out)