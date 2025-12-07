import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Xavier uniform initialization
        limit = 1 / np.sqrt(in_channels * np.prod(self.kernel_size))
        self.weights = np.random.uniform(
            -limit, limit, (out_channels, in_channels, *self.kernel_size)
        )
        self.bias = np.random.uniform(
            -limit, limit, (out_channels,)
            ) if bias else np.zeros(out_channels)

    def _pad_input(self, x):
        if self.padding == 0:
            return x
        return np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

    def forward(self, x):
        """
        x: shape (batch_size, in_channels, height, width)
        returns: (batch_size, out_channels, out_height, out_width)
          """
        x = np.array(x)
        batch_size, in_ch, in_h, in_w = x.shape
        kH, kW = self.kernel_size

        # Compute output shape
        out_h = (in_h - 2*self.padding - kH) // self.stride + 1
        out_w = (in_w - 2*self.padding - kW) // self.stride + 1

        # Pad input
        x_padded = self._pad_input(x)
        
        # Initialize output
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        # Convolution operation
        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(self.in_channels):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * self.stride
                            w_start = w * self.stride
                            patch = x_padded[b, i, h_start:h_start+kH, w_start:w_start+kW]
                            out[b, o, h, w] += np.sum(patch * self.weights[o, i])
                out[b, o] += self.bias[o]
        
        return out
