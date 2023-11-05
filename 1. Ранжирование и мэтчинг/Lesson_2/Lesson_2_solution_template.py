from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    # допишите ваш код здесь
    pass


def compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    pass


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    # допишите ваш код здесь
    pass


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    # допишите ваш код здесь
    pass


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # допишите ваш код здесь
    pass


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    pass


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    # допишите ваш код здесь
    pass


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    pass
