import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from sklearn.metrics import classification_report


def load_dataset(folder, shuffle: bool = False):
    """加载数据集

    Returns:
        inputs: (B, C, H, W)
        targets: (B, )
    """

    inputs = []
    targets = []
    for num in range(10):
        for img_path in Path(folder, str(num)).iterdir():
            inputs.append(np.asarray(PIL.Image.open(img_path)))
            targets.append(num)

    inputs = np.stack(inputs, 0, dtype=np.float32) / 255
    targets = np.stack(targets, 0, dtype=np.int32)

    if len(inputs.shape) == 3:
        inputs = inputs[:, None, :, :]  # (B, C, H, W)

    if shuffle:
        rnd_index = np.arange(inputs.shape[0])
        for _ in range(10):
            np.random.shuffle(rnd_index)
        inputs = inputs[rnd_index]
        targets = targets[rnd_index]

    return inputs, targets


class NetworkLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray, last_x_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ParamLayer(NetworkLayer):
    def update(self, lr: float) -> None:
        raise NotImplementedError


class LinearLayer(ParamLayer):
    """线性层"""

    def __init__(self, d1: int, d2: int) -> None:
        self.w = np.random.random((d1, d2)) * 2 - 1
        self.b = np.random.random((1, d2)) * 2 - 1

        self.w_grad = np.zeros_like(self.w)
        self.b_grad = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (B, d1)

        Returns:
            x: (B, d2)
        """

        return x @ self.w + self.b

    def backward(self, x: np.ndarray, last_x_grad: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (B, d1)
            last_grad: (B, d2)

        Returns:
            x_grad: (B, d1)
        """

        self.b_grad = last_x_grad.sum(0, keepdims=True)
        self.w_grad = x.T @ last_x_grad

        x_grad = last_x_grad @ self.w.T

        return x_grad

    def update(self, lr: float) -> None:
        self.w -= lr * self.w_grad
        self.b -= lr * self.b_grad


class ConvolutionLayer(ParamLayer):
    """步长为 1 的卷积层"""

    def __init__(self, c1: int, c2: int, k1: int, k2: int) -> None:
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k1: 卷积核高
            k2: 卷积核宽
        """

        self.k = np.random.random((c2, c1, k1, k2)) * 2 - 1
        self.b = np.random.random((1, c2, 1, 1)) * 2 - 1
        self.k_grad = np.zeros_like(self.k)
        self.b_grad = np.zeros_like(self.b)

    def _unfold_k(self, h: int, w: int, c1: int, c2: int, k1: int, k2: int, o1: int, o2: int) -> np.ndarray:
        """展开卷积核 k, 能够与展平的多通道图像直接相乘

        Returns:
            k: (c2 * o1 * o2, c1 * H * W)
                有 c2 个 (o1 * o2, c1 * H * W) 的展平卷积核, 每一个核大小是 c1 * H * W, 将在一张 c1 通道的一维图片上进行 o1 * o2 次卷积
        """

        ridx = np.arange(o1 * o2).reshape(-1, 1).repeat(k1 * k2, 1)

        cidx1 = np.arange(o2).reshape(1, -1).repeat(k1 * k2, 1).repeat(o1, 0).reshape(o1 * o2, k1 * k2)
        cidx2 = np.arange(k2).reshape(1, -1).repeat(o1 * o2 * k1, 0).reshape(o1 * o2, k1 * k2)
        cidx3 = np.arange(0, k1 * w, w).reshape(1, -1).repeat(k2, 1).repeat(o1 * o2, 0)
        cidx4 = np.arange(0, o1 * w, w).reshape(-1, 1).repeat(o2, 0).repeat(k1 * k2, 1)
        cidx = cidx1 + cidx2 + cidx3 + cidx4

        k = np.zeros((c2, c1, o1 * o2, h * w))
        k[:, :, ridx, cidx] = self.k.reshape(c2, c1, 1, k1 * k2).repeat(o1 * o2, 2)
        k = k.transpose(0, 2, 1, 3).reshape(c2 * o1 * o2, c1 * h * w)

        return k

    def _unfold_x(self, x: np.ndarray, c1: int, k1: int, k2: int, o1: int, o2: int) -> np.ndarray:
        """展开输入 x, 能够与展平的卷积核直接相乘

        Args:
            x: (B, c1, H, W)

        Returns:
            x: (B, o1 * o2, c1 * k1 * k2)
                每张图片将与 c1 通道的一维卷积核进行 o1 * o2 次卷积
        """

        ridx_r = np.arange(k1).reshape(1, k1).repeat(k2, 1).repeat(o1 * o2, 0)
        ridx_c = np.arange(o1).reshape(o1, 1).repeat(o2, 0).repeat(k1 * k2, 1)
        ridx = ridx_r + ridx_c

        cidx_r = np.arange(k2).reshape(1, k2).repeat(k1, 0).reshape(1, -1).repeat(o1 * o2, 0)
        cidx_c = np.arange(o2).reshape(1, o2).repeat(o1, 0).reshape(-1, 1).repeat(k1 * k2, 1)
        cidx = cidx_r + cidx_c

        x = x[:, :, ridx, cidx].transpose(0, 2, 1, 3).reshape(-1, o1 * o2, c1 * k1 * k2)
        return x

    def forward(self, x: np.ndarray):
        """
        Args:
            x: (B, c1, H, W)

        Returns:
            x: (B, c2, o1, o2)
        """
        _, _, h, w = x.shape
        c2, c1, k1, k2 = self.k.shape
        o1 = h - k1 + 1
        o2 = w - k2 + 1

        # 展开卷积核
        output = x.reshape(-1, c1 * h * w) @ self._unfold_k(h, w, c1, c2, k1, k2, o1, o2).T
        output = output.reshape(-1, c2, o1, o2) + self.b
        return output

        # # 展开输入 (B, o1 * o2, c1 * k1 * k2) @ (c1 * k1 * k2, c2) = (B, o1 * o2, c2)
        # output = self._unfold_x(x, c1, k1, k2, o1, o2) @ self.k.reshape(c2, c1 * k1 * k2).T
        # output = output.transpose(0, 2, 1).reshape(-1, c2, o1, o2) + self.b
        # return output

    def backward(self, x: np.ndarray, last_x_grad: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (B, c1, H, W)
            last_x_grad: (B, c2, o1, o2)

        Returns:
            x_grad: (B, c1, H, W)
        """

        _, _, h, w = x.shape
        c2, c1, k1, k2 = self.k.shape
        o1 = h - k1 + 1
        o2 = w - k2 + 1

        unfold_x = self._unfold_x(x, c1, k1, k2, o1, o2)
        unfold_k = self._unfold_k(h, w, c1, c2, k1, k2, o1, o2)

        self.b_grad = last_x_grad.sum((0, 2, 3), keepdims=True)
        self.k_grad = unfold_x.reshape(-1, c1 * k1 * k2).T @ last_x_grad.transpose(0, 2, 3, 1).reshape(-1, c2)
        self.k_grad = self.k_grad.transpose().reshape(c2, c1, k1, k2)

        x_grad = last_x_grad.reshape(-1, c2 * o1 * o2) @ unfold_k
        x_grad = x_grad.reshape(-1, c1, h, w)
        return x_grad

    def update(self, lr: float):
        self.k -= lr * self.k_grad
        self.b -= lr * self.b_grad


class MaxPoolingLayer(NetworkLayer):
    """步长与核大小相同的最大池化层"""

    def __init__(self, k1: int, k2: int) -> None:
        self.k1 = k1
        self.k2 = k2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            x: (B, C, H / k1, W / k2)
        """

        _, c, h, w = x.shape
        k1 = self.k1
        k2 = self.k2
        o1 = h // k1
        o2 = w // k2

        x = x.reshape(-1, o1, k1, o2, k2).transpose(0, 1, 3, 2, 4).reshape(-1, k1 * k2)
        ridx = np.arange(x.shape[0])
        cidx = np.argmax(x, axis=-1)

        x = x[ridx, cidx].reshape(-1, c, o1, o2)
        return x

    def backward(self, x: np.ndarray, last_x_grad: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (B, C, H, W)
            last_x_grad: (B, C, H / k1, W / k2)

        Returns:
            x_grad: (B, C, H, W)
        """

        _, c, h, w = x.shape
        k1 = self.k1
        k2 = self.k2
        o1 = h // k1
        o2 = w // k2

        x = x.reshape(-1, o1, k1, o2, k2).transpose(0, 1, 3, 2, 4).reshape(-1, k1 * k2)
        ridx = np.arange(x.shape[0])
        cidx = np.argmax(x, axis=-1)

        x_grad: np.ndarray = np.zeros_like(x).reshape(-1, k1 * k2)
        x_grad[ridx, cidx] = last_x_grad.reshape(-1)
        x_grad = x_grad.reshape((-1, o1, o2, k1, k2)).transpose((0, 1, 3, 2, 4)).reshape((-1, c, h, w))
        return x_grad


class ReLULayer(NetworkLayer):
    """ReLU"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """"""
        return x * (x > 0)

    def backward(self, x: np.ndarray, last_x_grad: np.ndarray) -> np.ndarray:
        """"""
        return last_x_grad * (x > 0)


class FlattenLayer(NetworkLayer):
    """Flatten"""

    def __init__(self, start: int = 1, end: int = -1) -> None:
        self.start = start
        self.end = end

    def forward(self, x: np.ndarray) -> np.ndarray:
        """"""

        shape = x.shape
        shapen_len = len(shape)
        start = (self.start + shapen_len) % shapen_len
        end = (self.end + shapen_len) % shapen_len

        return x.reshape(shape[:start] + (-1, ) + shape[end + 1:])

    def backward(self, x: np.ndarray, last_x_grad: np.ndarray) -> np.ndarray:
        """"""

        return last_x_grad.reshape(x.shape)


class CrossEntropyLoss(NetworkLayer):
    """"""

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Args:
            x: (B, C)
            y: (B, )

        Returns:
            loss: float number
        """
        x = x - x.max(-1, keepdims=True)
        loss = np.log(np.exp(x).sum(-1)) - x[np.arange(x.shape[0]), y]

        return float(loss.mean())

    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (B, C)
            y: (B, )

        Returns:
            x_grad: (B, C)
        """
        exp_x = np.exp(x - x.max(-1, keepdims=True))
        x_grad = exp_x / exp_x.sum(-1, keepdims=True)
        x_grad[np.arange(x.shape[0]), y] -= 1
        x_grad /= x.shape[0]

        return x_grad


class ConvolutionNeuralNetwork:
    """
    Conv -> Pool -> ReLU -> Conv -> Pool -> ReLU -> Flatten -> Linear -> CrossEntropy
    """

    def __init__(self) -> None:
        self.layers = [
            ConvolutionLayer(1, 4, 5, 5),  # 24 * 24
            MaxPoolingLayer(2, 2),  # 12 * 12
            ReLULayer(),
            ConvolutionLayer(4, 16, 3, 3),  # 10 * 10
            MaxPoolingLayer(2, 2),  # 5 * 5
            ReLULayer(),
            FlattenLayer(),
            LinearLayer(16 * 5 * 5, 10)
        ]
        self.loss_func = CrossEntropyLoss()

        self.layer_inputs = []

    def forward(self, x: np.ndarray, keepgrad: bool = False) -> np.ndarray:
        """
        Args:
            x: (B, C_ch, H, W), images
            keepgrad: whether keep temp layer inputs

        Returns:
            x: (B, C_cls), logits
        """

        for layer in self.layers:
            if keepgrad:
                self.layer_inputs.append(x)
            x = layer.forward(x)
        return x

    def backward(self, last_x_grad: np.ndarray) -> np.ndarray:
        """
        Args:
            last_x_grad: (B, C_cls), computed by loss function

        Returns:
            last_x_grad: (B, C_ch, H, W)
        """

        for x, layer in zip(self.layer_inputs[::-1], self.layers[::-1]):
            last_x_grad = layer.backward(x, last_x_grad)
        return last_x_grad

    def update(self, lr: float) -> None:
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                layer.update(lr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (B, C_ch, H, W), images

        Returns:
            x: (B, C_cls), probs by softmax
        """

        x = self.forward(x)
        exp_x = np.exp(x - x.max(-1, keepdims=True))
        outputs = exp_x / exp_x.sum(-1, keepdims=True)
        return outputs

    def train(self, train_x: np.ndarray, train_y: np.ndarray, batch_size: int, epochs: int, lr: float) -> float:
        """

        Returns:
            loss: mean loss for train_x
        """
        losses = []
        for i in range(0, train_x.shape[0], batch_size):
            inputs, targets = train_x[i:i+batch_size], train_y[i:i+batch_size]

            # forward
            logits = self.forward(inputs, True)
            loss = self.loss_func.forward(logits, targets)
            losses.append(loss)

            # backward
            last_x_grad = self.loss_func.backward(logits, targets)
            self.backward(last_x_grad)

            # update
            self.update(lr)

            # clear temp values
            self.layer_inputs.clear()

        return sum(losses) / len(losses)

    def test(self, test_x: np.ndarray, test_y: np.ndarray, batch_size: int) -> float:
        """
        Returns:
            loss: mean loss for test_x
        """

        losses = []
        for i in range(0, test_x.shape[0], batch_size):
            inputs, targets = test_x[i:i+batch_size], test_y[i:i+batch_size]

            # forward
            logits = self.forward(inputs)
            loss = self.loss_func.forward(logits, targets)
            losses.append(loss)

        return sum(losses) / len(losses)


if __name__ == "__main__":
    np.random.seed(1234)

    train_x, train_y = load_dataset("./cv-data/train", True)
    test_x, test_y = load_dataset("./cv-data/test")

    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    batch_size = 100
    epochs = 200
    lr = 0.1
    cnn = ConvolutionNeuralNetwork()

    train_losses = []
    test_losses = []
    print("=============== Begin Train ===============")
    start_time = time.time()
    for i in range(epochs):
        train_loss = cnn.train(train_x, train_y, batch_size, epochs, lr)
        test_loss = cnn.test(test_x, test_y, batch_size)
        print(f"Epoch: {i + 1:3d} Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f}")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    time_elapsed = time.time() - start_time
    print(f"=============== End Train: {time_elapsed / 60:.2f} min ===============")

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    # 绘制损失变化曲线, 忽略第一轮损失
    plt.plot(np.arange(train_losses.shape[0]-1), train_losses[1:], label="train")
    plt.plot(np.arange(test_losses.shape[0]-1), test_losses[1:], label="test")
    plt.legend()
    plt.savefig("loss.png")

    y_true = train_y
    y_pred = cnn.predict(train_x).argmax(-1)
    report = classification_report(y_true, y_pred, digits=4)
    print("=============== Classification Report: Train ===============")
    print(report)

    y_true = test_y
    y_pred = cnn.predict(test_x).argmax(-1)
    report = classification_report(y_true, y_pred, digits=4)
    print("=============== Classification Report: Test ===============")
    print(report)
