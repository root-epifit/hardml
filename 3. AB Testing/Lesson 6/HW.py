import pandas as pd
import numpy as np
import datetime as dt

def calculate_linearized_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name, kappa=None
):
    """Вычисляет значение линеаризованной метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словарь с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит в
        полуинтервал, а дата окончания нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики
    kappa - float, коэффициент в функции линеаризации.
        Если None, то посчитать как ratio метрику по имеющимся данным.

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """

    # YOUR_CODE_HERE
    data = df.copy()
    data[date_name] = pd.to_datetime(data[date_name])
    mask = (period['begin'] <= data[date_name]) & (data[date_name] < period['end']) & (data[user_id_name].isin(list_user_id))
    
    data = data[mask]\
            .drop(columns=[date_name])\
            .groupby(user_id_name)\
            .agg(['sum','count'])\
            .reset_index()

    columns = [f"{a}_{b}" if user_id_name not in f"{a}" else user_id_name for a,b in data.columns]
    data.columns = columns

    x = data[f"{value_name}_sum"].to_numpy()
    y = data[f"{value_name}_count"].to_numpy()
    if kappa is None: kappa = x.sum()/y.sum()

    l = x - kappa * y

    data[metric_name] = l
    data = pd.DataFrame(data = np.array(list_user_id), columns=[user_id_name]).merge(data, how='outer', on=[user_id_name]).fillna(0)

    rc = pd.DataFrame()
    rc[user_id_name] = data[user_id_name]
    rc[metric_name]  = data[metric_name]

    return rc