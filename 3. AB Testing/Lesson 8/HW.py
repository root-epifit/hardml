import pandas as pd
import numpy as np
import datetime
import scipy.stats as stats
from scipy.stats import norm


class SequentialTester:
    def __init__(
        self, metric_name, time_column_name,
        alpha, beta, pdf_one, pdf_two
    ):
        """Создаём класс для проверки гипотезы о равенстве средних тестом Вальда.

        Предполагается, что среднее значение метрики у распределения альтернативной
        гипотезы с плотность pdf_two больше.

        :param metric_name: str, название стобца со значениями измерений.
        :param time_column_name: str, названия столбца с датой и временем измерения.
        :param alpha: float, допустимая ошибка первого рода.
        :param beta: float, допустимая ошибка второго рода.
        :param pdf_one: function, функция плотности распределения метрики при H0.
        :param pdf_two: function, функция плотности распределения метрики при H1.
        """
        self.metric_name = metric_name
        self.time_column_name = time_column_name
        self.alpha = alpha
        self.beta = beta
        self.pdf_one = pdf_one
        self.pdf_two = pdf_two
        
        # YOUR_CODE_HERE
        self.data_control = pd.DataFrame()
        self.data_pilot   = pd.DataFrame()

    def run_test(self, data_control, data_pilot):
        """Запускаем новый тест, проверяет гипотезу о равенстве средних.
        
        :param data_control: pd.DataFrame, данные контрольной группы.
        :param data_pilot: pd.DataFrame, данные пилотной группы.
        
        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных 
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        # YOUR_CODE_HERE
        
        # Сохраняем данные.Будут как reference
        if 0 == self.data_control.shape[0]:
            data_control[self.time_column_name] = pd.to_datetime(data_control[self.time_column_name])
            data_pilot[self.time_column_name]   = pd.to_datetime(data_pilot[self.time_column_name])
            self.data_control = pd.concat([self.data_control, data_control], axis=0)
            self.data_pilot   = pd.concat([self.data_pilot, data_pilot], axis=0)
        else:
            print(f"run_test: WARNING: Using pre-saved data: shape={self.data_control.shape}")
        
        return self._test_sequential_wald(
            data_one = self.data_control[self.metric_name].to_numpy(), 
            data_two = self.data_pilot[self.metric_name].to_numpy(),
            pdf_one = self.pdf_one,
            pdf_two = self.pdf_two,
            alpha = self.alpha,
            beta = self.beta)

    
    def add_data(self, data_control, data_pilot):
        """Добавляет новые данные, проверяет гипотезу о равенстве средних.
        
        Гарантируется, что данные новые и не дублируют ранее добавленные.
        
        :param data_control: pd.DataFrame, новые данные контрольной группы.
        :param data_pilot: pd.DataFrame, новые данные пилотной группы.
        
        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных 
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        # YOUR_CODE_HERE
        #print(f"Before: {st.data_control.shape=}, {st.data_pilot.shape}")
        self.data_control = self._add_data_2_df1(self.data_control, data_control, self.time_column_name)
        self.data_pilot   = self._add_data_2_df1(self.data_pilot, data_pilot, self.time_column_name)
        #print(f"After: {st.data_control.shape=}, {st.data_pilot.shape}")
 
        return self.run_test(self.data_control, self.data_pilot)

        
    # helpers
    def _add_data_2_df1(self, df1, df2, time_column_name) -> pd.DataFrame:
        """
        Добавляет в df1 новые данные из df2. "новые" - это те, у которых ts > df1_max_ts
        """
        df1_max_ts = df1[time_column_name].max()
        df2[time_column_name] = pd.to_datetime(df2[time_column_name])
        mask = (df2[time_column_name] > df1_max_ts)
        df1 = pd.concat([df1, df2[mask]],ignore_index=True).drop_duplicates()
        #print(df1)
        return df1

    def _test_sequential_wald(self, data_one, data_two, pdf_one, pdf_two, alpha, beta):
        """Последовательно проверяет отличие по мере поступления данных.
    
        pdf_one, pdf_two - функции плотности распределения при нулевой и альтернативной гипотезах
    
        Возвращает 1, если были найдены значимые отличия, иначе - 0. И кол-во объектов при принятии решения.
        """
        lower_bound = np.log(beta / (1 - alpha))
        upper_bound = np.log((1 - beta) / alpha)
    
        min_len = min([len(data_one), len(data_two)])
        data_one = data_one[:min_len]
        data_two = data_two[:min_len]
        delta_data = data_two - data_one
    
        pdf_one_values = pdf_one(delta_data)
        pdf_two_values = pdf_two(delta_data)
    
        z = np.cumsum(np.log(pdf_two_values / pdf_one_values))
    
        indexes_lower = np.arange(min_len)[z < lower_bound]
        indexes_upper = np.arange(min_len)[z > upper_bound]
        first_index_lower = indexes_lower[0] if len(indexes_lower) > 0 else min_len + 1
        first_index_upper = indexes_upper[0] if len(indexes_upper) > 0 else min_len + 1
    
        if first_index_lower < first_index_upper:
            return 0, first_index_lower + 1
        elif first_index_lower > first_index_upper:
            return 1, first_index_upper + 1
        else:
            return 0.5, min_len