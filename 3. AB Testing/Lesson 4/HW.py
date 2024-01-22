import numpy as np
import pandas as pd


def select_stratified_groups(data, strat_columns, group_size, weights=None, seed=None):
    """Подбирает стратифицированные группы для эксперимента.

    data - pd.DataFrame, датафрейм с описанием объектов, содержит атрибуты для стратификации.
    strat_columns - List[str], список названий столбцов, по которым нужно стратифицировать.
    group_size - int, размеры групп.
    weights - dict, словарь весов страт {strat: weight}, где strat - либо tuple значений элементов страт,
        например, для strat_columns=['os', 'gender', 'birth_year'] будет ('ios', 'man', 1992), либо просто строка/число.
        Если None, определить веса пропорционально доле страт в датафрейме data.
    seed - int, исходное состояние генератора случайных чисел для воспроизводимости
        результатов. Если None, то состояние генератора не устанавливается.

    return (data_pilot, data_control) - два датафрейма того же формата, что и data
        c пилотной и контрольной группами.
    """
    # YOUR_CODE_HERE
#
# Variant 2
#
    np.random.seed(seed)

    data_pilot = pd.DataFrame(data=None, columns=data.columns)
    data_control=pd.DataFrame(data=None, columns=data.columns)

    if weights is None:
        weights = data[strat_columns].value_counts()/len(data)

    # Пересчитываем веса в штуки и преобразуем {строка:число} в {(строка):число}
    #print(f"Пересчитываем веса в штуки")
    sizes = dict()
    for stratum, weight in weights.items():
        #print(f"{stratum=}, check={isinstance(stratum, str)}, {weight=}")
        size = np.array(weight * group_size).round().astype('int')
        if isinstance(stratum, (str, float, int)):
            # stratum - строка/number, преобразуем в iterable (tuple)
            key = tuple([stratum])
            sizes[key] = size
        else:
            sizes[stratum] = size 

    # Формируем группы
    for stratum, size in sizes.items():
        #print(f"\n{stratum=}, {size=}")
        mask = True
        for i, value in enumerate(stratum):
            #print(f"{i=}, {strat_columns[i]=}, {value=}")
            mask = mask & (data[strat_columns[i]] == value)
        df = data[mask]
        if (size != 0) and (len(df) >= 2*size):
            df = df.take(np.random.permutation(len(df))[:(2*size)])
            data_pilot = pd.concat([data_pilot, df[:size]])
            data_control = pd.concat([data_control, df[-size:]])
        else:
            # требуемый расчетный размер страты = 0, т.е. group_size слишком маленький
            # или
            # В data не хватает данных для формирования страты требуемого размера
            data_pilot = pd.DataFrame(data=None, columns=data.columns)
            data_control=pd.DataFrame(data=None, columns=data.columns)
            break;
        
    return data_pilot, data_control