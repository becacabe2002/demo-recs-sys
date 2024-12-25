from typing import List, Dict, Any
import numpy as np
from utils.crt import CrtEnc
from utils.new_gd_ds import GradDataStruct

class SecureOps:
    def __init__(self, crt_enc):
        self.crt_enc = crt_enc
        self.grad_struct = None

    def process_rating(self, enc_ratings, masks):
        '''
        Proccess masked and encrypted rating to create initial structure
        '''
        dec_ratings = []
        for enc_rating, mask in zip(enc_ratings, masks):
            ct = self.crt_enc.dec_crt(enc_rating['ciphertext'])
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
                'grad_struct': self.grad_struct.__dict__,
                'processed_ratings': dec_ratings
                }

    def fixed_point_compute(self, vecs, op):
        '''
        Fixed point operation on encrypted vectors
        '''
        if op == 'agg_users':
            res = self.grad_struct.agg_u(vecs['data'])
        elif op == 'agg_items':
            res = self.grad_struct.agg_v(vecs['data'])
        elif op == 'recon_users':
            res = self.grad_struct.rec_u(vecs['data'])
        elif op == 'recon_items':
            res = self.grad_struct.rec_v(vecs['data'])
        else:
            raise ValueError(f"Unknown operation: {op}")
        
        enc_res = self.crt_enc.enc_crt(res)
        return {
                'result': enc_res
                }

    def check_convergence(self, grads, thres):
        user_norm = np.linalg.norm(grads['user_gradients'])
        item_norm = np.linalg.norm(grads['item_gradients'])
        return user_norm < thres and item_norm < thes
