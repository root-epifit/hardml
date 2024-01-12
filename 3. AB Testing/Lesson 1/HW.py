import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """
    # YOUR_CODE_HERE
    def check_ttest(a, b, alpha=0.05):
        """Тест Стьюдента. Возвращает 1, если отличия значимы."""
        _, pvalue = ttest_ind(a, b)
        return int(pvalue < alpha)

    B = n_iter
    np.random.seed(seed=seed)
    
    a = df_pilot_group[metric_name].to_numpy()
    b = df_control_group[metric_name].to_numpy()
    a_bootstrap = np.random.choice(a, size=(len(a), B))
    b_bootstrap = np.random.choice(b, size=(len(b), B))

    results = np.array([check_ttest(a_bootstrap[:,i], b_bootstrap[:,i], alpha=alpha) for i in range(n_iter)])
    #print(f"{np.sum(results)/results.size}, {np.mean(results)}")
    #return np.sum(results)/results.size
    return np.mean(results)


def estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибки второго рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """
    # YOUR_CODE_HERE
    def check_ttest(a, b, alpha=0.05):
        """Тест Стьюдента. Возвращает 1, если отличия значимы."""
        _, pvalue = ttest_ind(a, b)
        return int(pvalue < alpha)

    B = n_iter
    np.random.seed(seed=seed)
    a = df_pilot_group[metric_name].to_numpy()
    b = df_control_group[metric_name].to_numpy()
    
    a_bootstrap = np.random.choice(a, size=(len(a), B))
    b_bootstrap = np.random.choice(b, size=(len(b), B))
    
    result=dict()
    for effect in effects:
        a_bootstrap_effect=a_bootstrap*effect
        err_2 = np.array([check_ttest(a_bootstrap_effect[:,i], b_bootstrap[:,i], alpha=alpha) for i in range(n_iter)])
        
        #err_2 = np.array([check_ttest(a_bootstrap[:,i]*effect, b_bootstrap[:,i], alpha=alpha) for i in range(n_iter)])

        result[effect]=np.mean(abs(err_2-1))
        print(f"{effect=}, {result[effect]}")
    
    return result