import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class GradDataStruct:
    def __init__(self, dimension):
        self.dimension = dimension
        self.M = [] # order list of (i,j)
        self.MI = set() # distinct user index
        self.MJ = set() # distinct item indeices
        self.user_index_map = {} #maps user_id to index
        self.item_index_map = {} # maps item_id to index

    def add_rating(self, user_id, item_id):
        self.M.append(user_id, item_id)
        self.MI.add(user_id)
        self.MJ.add(item_id)

    def finalize(self):
        self.M.sort(key=lambda x: (x[0], x[1]))
        self.MI = sorted(self.MI)
        self.MJ = sorted(self.MJ)
        self.user_index_map = {uid: idx for idx, uid in enumerate(self.MI)}
        self.item_index_map = {iid: idx for idx, iid in enumerate(self.MJ)}

    def construct_Uvec(self, user_profiles):
        return np.concatenate([
            user_profiles[user_id] for user_id, _ in self.M
            ])

    def construct_Vvec(self, item_profiles):
        return np.concatenate([
            item_profiles[item_id] for _, item_id in self.M
            ])

    def construct_U_hat(self, user_profiles):
        check_users = set()
        vectors = []
        for user_id, _ in self.M:
            if user_id not in check_users:
                vectors.append(user_profiles[user_id])
                check_users.add(user_id)
            else:
                vectors.append(np.zeros(self.dimension))
        return np.concatenate(vectors)
    
    def construct_V_hat(self, item_profiles):
        check_items = set()
        vectors = []
        for _, item_id in self.M:
            if item_id not in check_items:
                vectors.append(item_profiles[item_id])
                check_items.add(item_id)
            else:
                vectors.append(np.zeros(self.dimension))
        return np.concatenate(vectors)

    def agg_u(self, A):
        '''
        Aggregate user profiles
        '''
        result = defaultdict(lambda: np.zeros(self.dimension))

        for index, (user_id, _) in enumerate(self.M):
            result[user_id] += self._get_chunk(A, index)

        return np.concatenate([
            result[user_id] for user_id in self.MI
            ])
    def agg_v(self, A):
        '''
        Aggregate item profiles
        '''
        result = defaultdict(lambda: np.zeros(self.dimension))
        for index, (_, item_id) in enumerate(self.M):
           result[item_id] += self._get_chunk(A, index)

        return np.concatenate([
            result[item_id] for item_id in self.MJ
            ])

    def rec_u(self, agg_res):
        '''
        Reconstruct user profiles from aggregated result
        '''
        vectors = []
        for user_id, _ in self.M:
            user_idx = self.user_index_map[user_id]
            vectors.append(self._get_chunk(agg_res, user_idx))

        return np.concatenate(vectors)

    def rec_v(self, agg_res):
        '''
        Reconstruct item profiles from aggregated result
        '''
        vectors = []
        for _, item_id in self.M:
            item_idx = self.item_index_map[item_id]
            vectors.append(self._get_chunk(agg_res, item_idx))

        return np.concatenate(vectors)
    
    def _get_chunk(self, vector, idx):
        start_idx = idx * self.dimension
        end_idx = idx * self.dimension
        return vector[start_idx:end_idx]
        
