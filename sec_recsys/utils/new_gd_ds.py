import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class GradDataStruct:
    def __init__(self, dimension):
        self.dimension = dimension
        self.M = [] # order list of (i,j)
        self.MI = set() # distinct user index
        self.MJ = set() # distinct item indeices
        self.user_set = set()
        self.item_set = set()
        self.user_index_map = {} #maps user_id to index
        self.item_index_map = {} # maps item_id to index

    def add_rating(self, user_id, item_id):
        self.M.append((user_id, item_id))
        self.user_set.add(user_id)
        self.item_set.add(item_id)

    def finalize(self):
        self.M.sort(key=lambda x: (x[0], x[1]))
        self.MI = sorted(list(self.user_set))
        self.MJ = sorted(list(self.item_set))
        self.user_index_map = {uid: idx for idx, uid in enumerate(self.MI)}
        self.item_index_map = {iid: idx for idx, iid in enumerate(self.MJ)}
        self.user_set.clear()
        self.item_set.clear()

    def get_u_pos(self, user_id):
        return self.user_index_map[user_id]
    
    def get_i_pos(self, item_id):
        return self.item_index_map[item_id]

    def construct_bold_U(self, user_profiles):
        return np.concatenate([
            user_profiles[user_id] for user_id, _ in self.M
        ])

    def construct_bold_V(self, item_profiles):
        return np.concatenate([
            item_profiles[item_id] for _, item_id in self.M
        ])

    def construct_U_hat(self, user_profiles):
        '''
        = u_i if i appears first time in M
        = d-length zero vector otherwise
        '''
        check_users = set()
        vecs = []
        for user_id, _ in self.M:
            if user_id not in check_users:
                user_idx = self.get_u_pos(user_id)
                vecs.append(user_profiles[user_idx])
                check_users.add(user_id)
            else:
                vecs.append(np.zeros(self.dimension))
        return np.vstack(vecs)
    
    def construct_V_hat(self, item_profiles):
        '''
        = v_j if j appears first time in M
        = d-length zero vector otherwise
        '''
        check_items = set()
        vecs = []
        for _, item_id in self.M:
            if item_id not in check_items:
                item_idx = self.get_i_pos(item_id)
                vecs.append(item_profiles[item_idx])
                check_items.add(item_id)
            else:
                vecs.append(np.zeros(self.dimension))
        return np.vstack(vecs)

    def agg_u(self, A):
        '''
        Aggregate user profiles: sums vectors for each unique user in MI
        '''
        result = {user_id: np.zeros_like(A[0]) for user_id in self.MI}
        for idx, (user_id, _) in enumerate(self.M):
            result[user_id] += A[idx]       
        return np.vstack(
                [result[user_id] for user_id in self.MI
            ])

    def agg_v(self, A):
        '''
        Aggregate item profiles: sums vectors for each unique item in MJ
        '''
        result = {item_id: np.zeros(self.dimension) for item_id in self.MJ}
        for idx, (_, item_id) in enumerate(self.M):
            result[item_id] += A[idx]
        return np.vstack([
            result[item_id] for item_id in self.MJ
            ])

    def rec_u(self, agg_res):
        '''
        Reconstruct user profiles from aggregated result
        '''
        vecs = []
        for user_id, _ in self.M:
            user_idx = self.user_index_map[user_id]
            vecs.append(agg_res[user_idx])

        return np.vstack(vecs)

    def rec_v(self, agg_res):
        '''
        Reconstruct item profiles from aggregated result
        '''
        vecs = []
        for _, item_id in self.M:
            item_idx = self.item_index_map[item_id]
            vecs.append(agg_res[item_idx])

        return np.vstack(vecs)
    
    # def _get_chunk(self, vector, idx):
    #     start_idx = idx * self.dimension
    #     end_idx = idx * self.dimension
    #     return vector[start_idx:end_idx]
    #
