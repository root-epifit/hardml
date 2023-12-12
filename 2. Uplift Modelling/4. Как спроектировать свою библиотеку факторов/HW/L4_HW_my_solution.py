# импорты для калкеров
import datetime
import dask.dataframe as dd
import featurelib as fl

# Импорты для трансформеров
import category_encoders as cat
import pandas as pd
import sklearn.base as skbase

#
# Калкеры
#
class DayOfWeekReceiptsCalcer(fl.DateFeatureCalcer):
    name = 'day_of_week_receipts'
    keys = ['client_id']

    def __init__(self, delta, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        receipts['day_of_week'] = receipts.transaction_datetime.dt.weekday.astype('category').cat.as_known()

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

        result = (
            receipts
            .loc[mask, ['client_id', 'transaction_id', 'day_of_week']]
            .drop_duplicates()
            .pivot_table(index='client_id', columns='day_of_week', values='transaction_id', aggfunc='count')
        )
        result.columns = [f'purchases_count_dw{col}__{self.delta}d' for col in result.columns]

        return result.reset_index()


class FavouriteStoreCalcer(fl.DateFeatureCalcer):
    name = 'favourite_store'
    keys = ['client_id']

    def __init__(self, delta, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)
        receipts = receipts[mask]

        trans_cnt = receipts.groupby(['client_id', 'store_id'])['transaction_id'].count().reset_index()
        trans_cnt_max = trans_cnt.groupby('client_id')['transaction_id'].max().reset_index()
        trans_filtered = trans_cnt.merge(trans_cnt_max)
        client_best_store = trans_filtered.groupby('client_id')['store_id'].max().reset_index()

        return client_best_store.rename(columns={'store_id': f'favourite_store_id__{self.delta}d'})
    

#
# Трансформеры
#
class ExpressionTransformer(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, expression, col_result):
        self.expression = expression
        self.col_result = col_result

    def fit(self, *args, **kwargs):
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        data[self.col_result] = eval(self.expression.format(d='data'))

        return data


class LOOMeanTargetEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, col_categorical, col_target, col_result):
        self.col_categorical = col_categorical
        self.col_target = col_target
        self.col_result = col_result
        self.encoder = cat.LeaveOneOutEncoder()

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        X_train = data[[self.col_categorical]]
        y_train = data[self.col_target] if self.col_target in data.columns else None
        self.encoder.fit(X_train, y_train)

        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        X_test = data[[self.col_categorical]]
        y_test = data[self.col_target] if self.col_target in data.columns else None
        data[self.col_result] = self.encoder.transform(X_test, y_test)[self.col_categorical]

        return data