from math import log2

from torch import Tensor, sort, stack

#  num_swapped_pairs — функция для расчёта количества неправильно упорядоченных пар 
# (корректное упорядочивание — от наибольшего значения в ys_true к наименьшему) или переставленных пар. 
# Не забудьте, что одну и ту же пару не нужно учитывать дважды.
def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    ys = stack((ys_pred, ys_true),-1)
    ys = Tensor(sorted(ys.numpy(), key = lambda x: x[0], reverse=True))
    ys_true_sorted = ys[:,1]

    wrong_sorted = 0
    for i, x in enumerate(ys_true_sorted[:-1]):
        for y in ys_true_sorted[i:]:
            wrong_sorted += int(x < y)

    return wrong_sorted


#compute_gain — вспомогательная функция для расчёта DCG и NDCG, рассчитывающая показатель Gain. 
#Принимает на вход дополнительный аргумент — указание схемы начисления Gain (gain_scheme).
#В лекции был приведён пример константного начисления, равного в точности оценке релевантности. 
#Необходимо реализовать как этот метод (при gain_scheme="const") начисления Gain, так и экспоненциальный 
#(gain_scheme="exp2"), рассчитываемый по формуле (2 r −1), 
#где r — реальная релевантность документа некоторому запросу. Логика здесь такова, что чем выше релевантность, тем ценнее объект, и темп роста “ценности" нелинейный — гораздо важнее отделить документ с релевантностью 5 от документа с релевантностью 3, нежели 3 от 1 (ведь они оба слабо релевантны).
#Функция принимает на вход единственное число (не тензор).

def compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    if gain_scheme == 'const':
         return y_value
    elif gain_scheme == 'exp2':
        return 2 ** y_value - 1

#dcg и ndcg — функции расчёта DCG и NDCG соответственно. 
#Принимают на вход дополнительный параметр gain_scheme, аналогичный таковому в функции compute_gain 
#(её необходимо использовать при расчётах этих метрик). 
#Для NDCG разрешается переиспользовать функцию расчёта DCG.
def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    ys = stack((ys_pred, ys_true),-1)
    ys = Tensor(sorted(ys.numpy(), key = lambda x: x[0], reverse=True))
    
    ys_true_sorted = ys[:,1]
    #print(ys_true_sorted)

    dcg=0.0
    for i,y in enumerate(ys_true_sorted.numpy()):
        gain= compute_gain(y, gain_scheme)
        k = i+1 # нумерация строк начинается с 1
        discount = log2(float(k+1))
        dcg += gain/discount
        #print(f'y={y}, i={i}, gain={gain}, discount={discount}, gain/discount={gain/discount}, dcg={dcg}')
                
    return dcg

def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    current_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    return current_dcg / ideal_dcg


#precission_at_k — функция расчёта точности в топ-k позиций для бинарной разметки 
#(в ys_true содержатся только нули и единицы). 
#Если среди лейблов нет ни одного релевантного документа (единицы), то необходимо вернуть -1. 
#Функция принимает на вход параметр k, указывающий на то, по какому количеству объектов необходимо произвести расчёт
#метрики. Учтите, что k может быть больше количества элементов во входных тензорах. 
#При реализации precission_at_k необходимо добиться того, что максимум функции в единице был достижим 
#при любом ys_true, за исключением не содержащего единиц 
#(попробуйте рассмотреть проблему на примере ранжирования поисковой выдачи, 
#где в разметке присутствует n<k релевантных документов). В силу этой особенности не рекомендуется переиспользовать
#имплементацию precission_at_k в других метриках настоящего ДЗ. 
#Можете также считать, что в ys_true содержатся все возможные единицы в датасете 
#(например, если система отбора кандидатов для ранжирования их пропустила, 
#то можно добавить такой объект c предсказанным значением −∞).

def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # Если в разметке нет релевантных документов, то возвращаем -1
    if int(ys_true.sum()) < 1:
        return -1.0
    
    ys = stack((ys_pred, ys_true),-1)
    ys = Tensor(sorted(ys.numpy(), key = lambda x: x[0], reverse=True))
    
    ys_pred_sotred = ys[:,0]
    ys_true_sorted = ys[:,1]
 
    k = min(k, ys_true_sorted.size()[0])
    k_subset = ys_true_sorted[:k]
    k = min(int(ys_true_sorted.sum()), k)
    return float(k_subset.sum()) / k
def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, indices = sort(ys_pred, descending=True)
    sorted_true = ys_true[indices]
    return float(1 / (sorted_true.argmax() + 1))


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    sorted_preds, indices = sort(ys_pred, descending=True)
    sorted_true = ys_true[indices]
    break_path = Tensor([(1 - p_break) ** i for i in range(1, ys_true.size()[0])])
    x = ((1 - sorted_true[:-1]).cumprod(dim=0) * break_path * sorted_true[1:]).sum()
    return float(x + sorted_true[0])

def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    n_positive = int(ys_true.sum())
    if not n_positive:
        return -1.0
    else:
        _, indices = sort(ys_pred, descending=True)
        sorted_true = ys_true[indices]
        tp = sorted_true.cumsum(0)
        pos = Tensor(list(range(1, tp.size()[0] + 1))).float()
        precision = tp.div(pos)
        ap = float(precision[sorted_true.bool()].sum() / n_positive)
        return ap