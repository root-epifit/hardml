import numpy as np
import pandas as pd


def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    """Оцениваем sample size для списка эффектов.

    df - pd.DataFrame, датафрейм с данными
    metric_name - str, название столбца с целевой метрикой
    effects - List[float], список ожидаемых эффектов. Например, [1.03] - увеличение на 3%
    alpha - float, ошибка первого рода
    beta - float, ошибка второго рода

    return - pd.DataFrame со столбцами ['effect', 'sample_size']    
    """
    # YOUR_CODE_HERE
    import scipy.stats as stats

    a = df[metric_name].values
    mu_a = np.mean(a)
    var_a = a.var()
    var_b = var_a

    t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
    z_scores_sum_squared = (t_alpha + t_beta) ** 2

    result = list()
    for effect in effects:
        epsilon = mu_a*(effect - 1)
        result.append([ effect,
                    int( np.ceil( z_scores_sum_squared * (var_a + var_b) / (epsilon ** 2) ) ) 
                  ])

    return pd.DataFrame(data=result, columns = ['effect', 'sample_size'])
