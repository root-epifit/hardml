import numpy as np
import pandas as pd


def calculate_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name
):
    """Вычисляет значение метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словарь с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит нужный
        полуинтервал, а дата окончание нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """
    # YOUR_CODE_HERE
    #print(f"{df.head(3)}\n\n{value_name=}, {user_id_name=}\n{list_user_id=}, \n{date_name=}, {period=}\n{metric_name=}")
    
    data = df.copy()
    agg_func='sum'
    
    data[date_name] = pd.to_datetime(df[date_name])
    mask = (period['begin'] <= data[date_name]) & (data[date_name] < period['end']) & (data[user_id_name].isin(list_user_id))
    
    data = data[mask]\
            .drop(columns=[date_name])\
            .groupby(user_id_name)\
            .agg(agg_func)\
            .reset_index()\
            .rename(columns={value_name:metric_name})
    
    data = pd.DataFrame({user_id_name: list_user_id})\
                .merge(data, how='outer', on=user_id_name)\
                .fillna(0)
            
    return data



def calculate_metric_cuped(
    df, value_name, user_id_name, list_user_id, date_name, periods, metric_name
):
    """Вычисляет метрики во время пилота, коварианту и преобразованную метрику cuped.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    periods - dict, словарь с датами начала и конца периода пилота и препилота.
        Пример, {
            'prepilot': {'begin': '2020-01-01', 'end': '2020-01-08'},
            'pilot': {'begin': '2020-01-08', 'end': '2020-01-15'}
        }.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами
        [user_id_name, metric_name, f'{metric_name}_prepilot', f'{metric_name}_cuped'],
        кол-во строк должно быть равно кол-ву элементов в списке list_user_id.
    """
    # YOUR_CODE_HERE
    # вычисляем значения метрики до пилота и во время пилота 
    prepilot = calculate_metric( df, value_name, user_id_name, list_user_id, date_name, periods['prepilot'], metric_name)
    pilot    = calculate_metric( df, value_name, user_id_name, list_user_id, date_name, periods['pilot'], metric_name)

    # cобираем метрики вместе, чтобы гарантировать правильную привязку к user_id
    data = pilot.merge(prepilot, how='inner', on = [user_id_name], suffixes=('', '_prepilot'))

    # вычисляем тету
    y = data[metric_name].values
    x = data[f'{metric_name}_prepilot'].values
    covariance = np.cov(y,x)[0, 1]
    variance = x.var()
    theta = covariance/variance

    data[f'{metric_name}_cuped'] = data[f'{metric_name}_cuped'] = y - theta * x

    return data