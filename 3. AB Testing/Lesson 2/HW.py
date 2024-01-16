import numpy as np
import pandas as pd


def calculate_sales_metrics(df, cost_name, date_name, sale_id_name, period, filters=None):
    """Вычисляет метрики по продажам.
    
    df - pd.DataFrame, датафрейм с данными. Пример
        pd.DataFrame(
            [[820, '2021-04-03', 1, 213]],
            columns=['cost', 'date', 'sale_id', 'shop_id']
        )
    cost_name - str, название столбца с стоимостью товара
    date_name - str, название столбца с датой покупки
    sale_id_name - str, название столбца с идентификатором покупки (в одной покупке может быть несколько товаров)
    period - dict, словарь с датами начала и конца периода пилота.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    filters - dict, словарь с фильтрами. Ключ - название поля, по которому фильтруем, значение - список значений,
        которые нужно оставить. Например, {'user_id': [111, 123, 943]}.
        Если None, то фильтровать не нужно.

    return - pd.DataFrame, в индексах все даты из указанного периода отсортированные по возрастанию, 
        столбцы - метрики ['revenue', 'number_purchases', 'average_check', 'average_number_items'].
        Формат данных столбцов - float, формат данных индекса - datetime64[ns].
    """
    # YOUR_CODE_HERE
        
    # фильтруем по date и фильтрам
    filters={} if filters is None else filters
    df['date'] = df['date'].astype('datetime64[ns]')
    
    mask = "(df['date'] >= period['begin']) & (df['date'] < period['end'])"

    for field,values in filters.items():
        mask_ = ''
        for value in values:
            mask_ = mask_ + f"|(df['{field}'] == {value})"   
        mask = mask + f'&({mask_[1::]})'    #  Убираем лидирующий '|'
    
    df_ = df[eval(mask)]
    
    # Группируем и считаем метрики
    df_agg = df_.groupby(['date','sale_id']).agg(
            revenue = ('cost', 'sum'),
            number_items = ('sale_id', 'count')
    ).reset_index('sale_id').groupby('date').agg(
            number_purchases = ('sale_id', 'count'),
            revenue = ('revenue', 'sum'),
            number_items = ('number_items', 'sum')
    ).reset_index('date')

    df_agg['average_check']        = df_agg['revenue']/df_agg['number_purchases']
    df_agg['average_number_items'] = df_agg['number_items']/df_agg['number_purchases']

    # заполняем дырки по датам пустыми записями
    df_fill = pd.DataFrame(data = np.arange(np.datetime64(period['begin']), np.datetime64(period['end'])), columns=['date'])
    df_final = df_fill.merge(df_agg, how='left', on='date')
    df_final = df_final.fillna(0)
    
    # Индексируем, сортируем и убираем лишние поля
    df_final.index = df_final['date']
    df_final = df_final.sort_index(axis=0)
    df_final = df_final.drop(columns=['date', 'number_items'], axis = 1)
    df_final = df_final.astype('float64')

    return df_final[['revenue', 'number_purchases', 'average_check', 'average_number_items']]


