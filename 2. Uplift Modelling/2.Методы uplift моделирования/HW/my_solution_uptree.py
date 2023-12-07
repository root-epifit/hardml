import numpy as np
from typing import Iterable, Tuple, List


class Node:
    def __init__(self, depth:int = 0, parent: object = None, node_type: str = 'Root'):
        #self.predicted_class = predicted_class
        self._n_items = 0
        self._depth = depth
        self._type = node_type
        self._ate = 0
        self._split_feat = None
        self._split_threshold = None
        self._parent = None
        self._left = None
        self._right = None

    def print(self):
        print("")
        print(f"node_id        = {self}")
        print(f"depth          = {self._depth}")
        print(f"type           = {self._type}")
        print(f"n_items        = {self._n_items}")
        print(f"ATE            = {self._ate}")
        print(f"split_feat     = {self._split_feat}")
        print(f"split_threshold= {self._split_threshold}")
        print(f"parent         = {self._parent}")
        print(f"left           = {self._left}")
        print(f"right          = {self._right}")
        
         
class UpliftTreeRegressor: 
    def __init__(
        self,
        max_depth: int = 3, # максимальная глубина дерева.
        min_samples_leaf: int = 1000, # минимальное необходимое число обучающих объектов в листе дерева.
        min_samples_leaf_treated: int = 300, # минимальное необходимое число обучающих объектов с T=1 в листе дерева.
        min_samples_leaf_control: int = 300, # минимальное необходимое число обучающих объектов с T=0 в листе дерева.
    ) -> None:
        # do something
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_leaf_treated = min_samples_leaf_treated
        self._min_samples_leaf_control = min_samples_leaf_control
        self._root_node = None
        
    def fit(
        self,
        X: np.ndarray,         # массив (n * k) с признаками.
        treatment: np.ndarray, # массив (n) с флагом воздействия.
        y: np.ndarray          # массив (n) с целевой переменной.
    ) -> None:
        # fit the model
        self._root_node = Node(depth=0)
        self._build_tree(self._root_node, X, y, treatment)
    
    def predict(self, X: np.ndarray) -> Iterable[float]:
        # compute predictions
        if self._root_node is None:
            raise ValueError(f'{self.__class__.__name__} is not fitted. Please use .fit() first.')
        return [self._predict(x) for x in X]
    
    def _predict(self, x: np.array) -> float:
        # Предсказание для отдного наблюдения x из X
        node = self._root_node
        while node._left:
            if x[node._split_feat] <= node._split_threshold:
                node = node._left
            else:
                node = node._right
        return node._ate
     
    def _get_threshold_options(self, column_values: List[float]) -> List[float]:
        unique_values = np.unique(column_values)
        if len(unique_values) > 10:
            percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])
        return np.unique(percentiles)

    def _is_satisfy_constraints(self, t) -> bool:
        min_samples, min_treated, min_control =  self._min_samples_leaf, self._min_samples_leaf_treated, self._min_samples_leaf_control
        samples = t.shape[0]
        treated = t[(t == 1)].shape[0]
        control = t[(t == 0)].shape[0]
        return (samples >= min_samples) and (treated >= min_treated) and (control >= min_control)
        
    def _best_split_on_axis(self,
                    X: np.array = None, 
                    y:np.array = None, 
                    t:np.array = None) -> Tuple[float, float]:
        """
        Возвращает лучший порог  и значение критерия для массива X - среза n-й фичи
        """
    
        # Получаем список возможных порогов
        threshold_options = self._get_threshold_options(X)
    
        # Итерируемся по списку возможных порогов и для каждого вычисляем delta_delta_p
        deltas = []
        for threshold in threshold_options:
            mask = (X <= threshold)

            X_left, y_left, t_left = X[mask], y[mask], t[mask]
            X_right, y_right, t_right = X[~mask], y[~mask], t[~mask]

            if self._is_satisfy_constraints(t_left) and self._is_satisfy_constraints(t_right):
                # разбиение удовлетворяет граничным условиям
                uplift_left = self._uplift(y_left, t_left)
                uplift_right = self._uplift(y_right, t_right)
                delta_delta_p = abs(uplift_left - uplift_right)
            else:
                # разбиение не удовлетворяет граничным условиям => исклюбчаем его из расмотрения
                # print(f"_best_split_on_axiы: разбиение не удовлетворяет граничным  условиям, treshold={threshold}")
                delta_delta_p = -1
            deltas.append(delta_delta_p)
 
        # находим порог, который дает макс.значение критерия
        deltas = np.array(deltas)
        delta_delta_p = deltas.max()
        threshold = threshold_options[(deltas == delta_delta_p)][0]
    
        return threshold, delta_delta_p


    def _best_split(self, X:np.ndarray, y:np.array, t:np.array) -> Tuple[int, float]:
        """
        Для матрицы наблюдений возвращает наилучшее разбиение, возвращает:
        - split_feat - индекс фактора по которому производить разбиение
        - split_threshold - порог
        """

        result = []
        n_feats = X.shape[1]
        for feature_idx in range(n_feats):
            threshold, delta_delta_p = self._best_split_on_axis(X[:,feature_idx], y, t)
            result.append([feature_idx, threshold, delta_delta_p])    
        result = np.array(result)

        mask = (result[:,2] == result.max(axis=0)[2])   # байтовая маска для строки с макс.значением delta_delta_p
        split_feat, split_threshold = result[mask][0].tolist()[:2] # отбираем строку
    
        return int(split_feat), split_threshold
        

    def _uplift(self, y, t) -> float:
        mask = (t == 1)
        y_c, t_c = y[~mask],  t[~mask]
        y_t, t_t = y[mask], t[mask]
    
        t_value = (y_t * t_t).sum()
        t_sum   = t_t.sum()

        c_value = (y_c * (1-t_c)).sum()
        c_sum   = (1-t_c).sum() 

        t_value_mean = 0 if t_sum == 0 else t_value/t_sum
        c_value_mean = 0 if c_sum == 0 else c_value/c_sum
         
        return t_value_mean - c_value_mean
    

    def _ate(self, y,t) -> float:
        y_mean_t = y[(t == 1)].mean()
        y_mean_c = y[(t == 0)].mean()
        return y_mean_t - y_mean_c
    
    
    def _build_tree(self,
                   node:Node,
                   X: np.ndarray = None, 
                   y:np.array = None, 
                   t:np.array = None) -> None:
    
        max_depth = self._max_depth
        next_node_depth = node._depth + 1
        
        node._n_items = X.shape[0]
        node._split_feat, node._split_threshold = self._best_split(X, y, t)
        node._ate = self._ate(y,t)

        mask = (X[:,node._split_feat] <= node._split_threshold)
                
        X_left, y_left, t_left = X[mask], y[mask], t[mask]
        X_right, y_right, t_right = X[~mask], y[~mask], t[~mask]
        
        if  next_node_depth <= max_depth:
            # Глубина дерева не превышена, можно создавать новый узел
            if self._is_satisfy_constraints(t_left):
                node._left = Node(depth=next_node_depth, parent=node, node_type='Left')
                self._build_tree(node._left, X_left, y_left, t_left)

            if self._is_satisfy_constraints(t_right):
                node._right = Node(depth=node._depth+1, parent=node, node_type='Right')
                self._build_tree(node._right, X_right, y_right, t_right)

        #node.print()
        
        return
