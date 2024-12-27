from typing import List, Dict, Any
import numpy as np
from utils.crt import CrtEnc
from utils.new_gd_ds import GradDataStruct
import tenseal as ts

class SecureOps:
    def __init__(self, key_manager, crt_enc):
        self.key_manager = key_manager
        self.crt_enc = crt_enc
        self.grad_struct = None

    def process_rating(self, masked_data):
        '''
        Proccess masked and encrypted rating to create initial structure
        '''
        if not self.grad_struct:
            self.grad_struct = GradDataStruct(dimension=20)
            for r in dec_ratings:
                self.grad_struct.add_rating(r['user_id'], r['item_id'])
            self.grad_struct.finalize()
        
        rating_vecs = []
        indicator_vecs = []

        for item in masked_data:
            # decrypt rating and indicator with AHE
            dec_rating = self.key_manager.decrypt_ahe(item['masked_rating'])
            dec_indicator = self.key_manager.decrypt_ahe(item['masked_indicator'])
        
            rating_dvec = np.zeros(self.grad_struct.dimension)
            rating_dvec[0] = -dec_rating
            rating_vecs.append(rating_dvec)

            indicator_dvec = np.full(self.grad_struct.dimension, dec_indicator)
            indicator_vecs.append(indicator_dvec)

        concatenated_ratings = np.concatenate(rating_vecs)
        concatenated_indicators = np.concatenate(indicator_vecs)

        # encrypt with CRT-CKKS
        enc_ratings = self.crt_enc.enc_crt(concatenated_ratings)
        enc_indicators = self.crt_enc.enc_crt(concatenated_indicators)

        return {
        'vectors':{
            'ratings': [ct.serialize().hex() for ct in enc_ratings],
            'indicators': [ct.serialize().hex() for ct in enc_indicators]
            }, 
        'grad_struct': {
            'dimension': self.grad_struct.dimension,
            'M': self.grad_struct.M,
            'MI': list(self.grad_struct.MI),  # Convert set to list
            'MJ': list(self.grad_struct.MJ),  # Convert set to list
            'user_index_map': self.grad_struct.user_index_map,
            'item_index_map': self.grad_struct.item_index_map
            }
        }

    def fixed_point_compute(self, vecs, op):
        '''
        Fixed point operation on encrypted vectors
        '''
        if not self.grad_struct or not self.grad_struct.MI: 
            return {'result': []}
        
        data = np.array([float(x) for x in vecs['data']])
    
        if op == 'agg_users':
            res = self.grad_struct.agg_u(data)
        elif op == 'agg_items':
            res = self.grad_struct.agg_v(data)
        elif op == 'recon_users':
            res = self.grad_struct.rec_u(data)
        elif op == 'recon_items':
            res = self.grad_struct.rec_v(data)
        else:
            raise ValueError(f"Unknown operation: {op}")
        
        enc_res = self.crt_enc.enc_crt(float(res[0] if len(res) > 0 else 0))
        return {
                'result': [ct.serialize().hex() for ct in enc_res]
                }

    def check_convergence(self, grads, thres):
        user_norm = np.linalg.norm(grads['user_gradients'])
        item_norm = np.linalg.norm(grads['item_gradients'])
        return bool(user_norm < thres and item_norm < thres)
