import numpy as np


class SingleNeuron(object):

    def __init__(self):
        self._w = 0         # 가중치 w
        self._b = 0         # 바이어스 b
        self._w_grad = 0
        self._b_grad = 0
        self._x = 0         # 입력값 x

    def set_params(self, w, b):
        """가중치와 바이어스를 저장합니다."""
        self._w = w
        self._b = b

    def forpass(self, x):
        """정방향 수식 w * x + b 를 계산하고 결과를 리턴합니다."""
        self._x = x
        _y_hat = self._w * self._x + self._b
        return _y_hat

    def backprop(self, err):
        """에러를 입력받아 가중치와 바이어스의 변화율을 곱하고 평균을 낸 후 감쇠된 변경량을 저장합니다."""
        m = len(self._x)
        self._w_grad = 0.1 * np.sum(err * self._x) / m
        self._b_grad = 0.1 * np.sum(err * 1) / m

    def update_grad(self):
        """계산된 파라메타의 변경량을 업데이트하여 새로운 파라메타를 셋팅합니다."""
        self.set_params(self._w + self._w_grad, self._b + self._b_grad)


