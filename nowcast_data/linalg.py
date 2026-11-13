"""极小的线性代数工具，避免引入第三方依赖。

只服务于本项目的小样本回归与不确定性模拟：矩阵规模通常小于 10×10。
"""
from __future__ import annotations


class SingularMatrix(ValueError):
    pass


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b:
        return []
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def identity(size: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元（部分主元）；奇异时抛出 SingularMatrix。"""
    n = len(a)
    if n == 0:
        return []
    if any(len(row) != n for row in a) or len(b) != n:
        raise ValueError("矩阵或向量维度不一致")
    augmented = [list(map(float, row)) + [float(b[i])] for i, row in enumerate(a)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise SingularMatrix("矩阵奇异")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= pivot_value
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]
    return [augmented[i][n] for i in range(n)]


def inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    if n == 0:
        return []
    columns = []
    for j in range(n):
        basis = [1.0 if i == j else 0.0 for i in range(n)]
        columns.append(solve(a, basis))
    return transpose(columns)


def cholesky(a: list[list[float]]) -> list[list[float]]:
    """下三角分解；失败时抛 SingularMatrix。"""
    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 1e-12:
                    raise SingularMatrix("协方差矩阵不是正定")
                lower[i][j] = value ** 0.5
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def sample_standard_normal(rng) -> float:
    """Box-Muller，rng 为 random.Random。"""
    import math

    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def sample_normal_vector(rng, mean: list[float], covariance: list[list[float]]) -> list[float]:
    """用 Cholesky 分解抽取多元正态；退化时退化为对角抽样。"""
    n = len(mean)
    if n == 0:
        return []
    try:
        lower = cholesky(covariance)
    except SingularMatrix:
        return [
            mean[i] + (max(covariance[i][i], 0.0) ** 0.5) * sample_standard_normal(rng)
            for i in range(n)
        ]
    z = [sample_standard_normal(rng) for _ in range(n)]
    return [mean[i] + sum(lower[i][k] * z[k] for k in range(i + 1)) for i in range(n)]


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("空序列没有分位数")
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


__all__ = [
    "SingularMatrix",
    "transpose",
    "matmul",
    "matvec",
    "identity",
    "solve",
    "inverse",
    "cholesky",
    "sample_standard_normal",
    "sample_normal_vector",
    "quantile",
]
