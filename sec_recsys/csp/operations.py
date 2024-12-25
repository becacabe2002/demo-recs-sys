from typing import List, Dict, Any
import numpy as np
from utils.crt import CrtEnc
from utils.new_gd_ds import GradDataStruct
import tenseal as ts

class SecureOps:
    def __init__(self, crt_enc):
        self.crt_enc = crt_enc
        self.grad_struct = GradDataStruct(dimension=20)

    def process_rating(self, enc_ratings, masks):
        '''
        Proccess masked and encrypted rating to create initial structure
        '''
        dec_ratings = []
        for enc_rating, mask in zip(enc_ratings, masks):
            cts = [ts.ckks_tensor_from(self.crt_enc.contexts[i], bytes.fromhex(ct_hex))
                for i, ct_hex in enumerate(enc_rating['ciphertext'])
                   ]
            ct = self.crt_enc.dec_crt(cts)
            um = (ct - mask['value']) % (1 << self.crt_enc.bit_size)
            dec_ratings.append({
                'user_id': enc_rating['user_id'],
                'item_id': enc_rating['item_id'],
                'rating': um
                })

        if not self.grad_struct:
            self.grad_struct = GradDataStruct(dimension=20)
            for r in dec_ratings:
                self.grad_struct.add_rating(r['user_id'], r['item_id'])
            self.grad_struct.finalize()

        return {
                'grad_struct': {
                    'dimension': self.grad_struct.dimension,
                    'M': self.grad_struct.M,
                    'MI': list(self.grad_struct.MI),  # Convert set to list
                    'MJ': list(self.grad_struct.MJ),  # Convert set to list
                    'user_index_map': self.grad_struct.user_index_map,
                    'item_index_map': self.grad_struct.item_index_map
                },
                'processed_ratings': dec_ratings
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
