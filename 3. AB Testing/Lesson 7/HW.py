import numpy as np
import pandas as pd
import hashlib


class ABSplitter:
    def __init__(self, count_slots, salt_one, salt_two):
        self.count_slots = count_slots
        self.salt_one = salt_one
        self.salt_two = salt_two

        self.slots = np.arange(count_slots)
        self.experiments = []
        self.experiment_to_slots = dict()
        self.slot_to_experiments = dict()

    def split_experiments(self, experiments):
        """Устанавливает множество экспериментов, распределяет их по слотам.

        Нужно определить атрибуты класса:
            self.experiments - список словарей с экспериментами
            self.experiment_to_slots - словарь, {эксперимент: слоты}
            self.slot_to_experiments - словарь, {слот: эксперименты}
        experiments - список словарей, описывающих пилот. Словари содержит три ключа:
            experiment_id - идентификатор пилота,
            count_slots - необходимое кол-во слотов,
            conflict_experiments - list, идентификаторы несовместных экспериментов.
            Пример: {'experiment_id': 'exp_16', 'count_slots': 3, 'conflict_experiments': ['exp_13']}
        return: List[dict], список экспериментов, которые не удалось разместить по слотам.
            Возвращает пустой список, если всем экспериментам хватило слотов.
        """
        self.experiments = experiments
        # YOUR_CODE_HERE
        slot_to_experiments, unassigned_experiments = self._match_pilot_slot_four(self.experiments, self.slots)

        experiment_to_slots={}
        for slot, experiments in slot_to_experiments.items():
            for experiment_id in experiments:
                if 0 == experiment_to_slots.get(experiment_id,0):
                    # если ключа ещё нет в словаре, то добавляем его
                    experiment_to_slots[experiment_id] = []
                experiment_to_slots[experiment_id].append(slot)
    
        self.experiment_to_slots = experiment_to_slots
        self.slot_to_experiments = slot_to_experiments
    
        return unassigned_experiments


    def process_user(self, user_id: str):
        """Определяет в какие эксперименты попадает пользователь.

        Сначала нужно определить слот пользователя.
        Затем для каждого эксперимента в этом слоте выбрать пилотную или контрольную группу.

        user_id - идентификатор пользователя.

        return - (int, List[tuple]), слот и список пар (experiment_id, pilot/control group).
            Example: (2, [('exp 3', 'pilot'), ('exp 5', 'control')]).
        """
        # YOUR_CODE_HERE
        user_experiments = []
        
        n_slots = len(self.slot_to_experiments.keys())
        user_slot = self._get_hash_modulo(f"{user_id}", n_slots, self.salt_one)

        for experiment_id in self.slot_to_experiments[user_slot]:
            group_name = {0:'control', 1:'pilot'}[self._get_hash_modulo(f"{user_id}{experiment_id}", 2, self.salt_two)]
            user_experiments.append( (experiment_id, group_name) )

            
        #print(f"{user_id}, {user_experiments}")
    
        return (user_slot, user_experiments)
        
    # ------------  хелперы --------------
    def _get_hash_modulo(self, value: str, n_slots: int, salt: int = 0):
        """Вычисляем остаток от деления: (hash(value) + salt) % modulo."""
        hash_value = int(hashlib.md5(str.encode(str(value) + f"{salt}")).hexdigest(), 16)
        user_slot = hash_value % n_slots
        return user_slot


    def _match_pilot_slot_four(self, pilots: list, slots: list):
        """Функция распределяет пилоты по слотам.
    
        pilots: список словарей, описывающих пилот. Содержит ключи:
            experiment_id - идентификатор пилота,
            count_slots - необходимое кол-во слотов,
            conflict_experiments - list, идентификаторы несовместных пилотов.
        slots: список с идентификаторами слотов.
    
        return: словарь соответствия на каких слотах какие пилоты запускаются, и список нераспределенных пилотов
            ( {slot_id: list_pilot_id, ...}, [{experiment_id: ....}, {...}] 
        """
        unassigned_pilots = []
        pilots = sorted(pilots, key=lambda x: len(x['conflict_experiments']), reverse=True)
        #print(f"{pilots=}")
    
        slot_to_pilot = {slot: [] for slot in slots}
        pilot_to_slot = {pilot['experiment_id']: [] for pilot in pilots}
        for pilot in pilots:
            if pilot['count_slots'] > len(slots):
                print(f'ERROR: pilot_id={pilot["experiment_id"]} needs too many slots.')
                unassigned_pilots.append(pilot)
                continue

            # найдём доступные слоты
            notavailable_slots = []
            for conflict_pilot_id in pilot['conflict_experiments']:
                notavailable_slots += pilot_to_slot[conflict_pilot_id]
            available_slots = list(set(slots) - set(notavailable_slots))
        
            if pilot['count_slots'] > len(available_slots):
                print(f'ERROR: experiment_id="{pilot["experiment_id"]}" not enough available slots.')
                unassigned_pilots.append(pilot)
                continue

            # shuffle - чтобы внести случайность, иначе они все упорядочены будут по номеру slot
            np.random.shuffle(available_slots)
            available_slots_orderby_count_pilot = sorted(
                available_slots,
                key=lambda x: len(slot_to_pilot[x]), reverse=True
            )
            pilot_slots = available_slots_orderby_count_pilot[:pilot['count_slots']]
            pilot_to_slot[pilot['experiment_id']] = pilot_slots
            for slot in pilot_slots:
                slot_to_pilot[slot].append(pilot['experiment_id'])
        return slot_to_pilot, unassigned_pilots