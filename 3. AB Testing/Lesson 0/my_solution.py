# Функция, которая будет вычислять доверительный интервал оценки параметра p
# распределения Бернулли
# при решении используйте приближения z(alfa/2)=1.96

import numpy as np

def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.

    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    """
    
    from math import sqrt

    z_alfa_1_2 = 1.96
    n = values.size
    mu = np.mean(values)
    sigma = np.std(values)/sqrt(n)

    return np.clip((mu - z_alfa_1_2*sigma, mu + z_alfa_1_2*sigma), 0,1)


if __name__ == "__main__":
    
    #np.random.seed(10)

    # Генерируем выборку размером size случ.величин из Бернулевского распределения вероятности с параметром p
    for p in [0.1, 0.5, 0.8]:
        print("")
        for size in [10, 30, 50, 100, 1000]:
            values=np.random.binomial(size=size, n=1, p=p)
            if p <=0.1 and size == 10: print(f"{values=}")
            left_bound, right_bound = get_bernoulli_confidence_interval(values=values)
            print(f"{p=}, {size=}, {left_bound=}, {right_bound=}")
            