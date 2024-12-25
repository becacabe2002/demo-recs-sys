import unittest
import json
import numpy as np
from csp_server import app
from utils.crt import CrtEnc

class TestCSPServer(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.app.testing = True
        self.crt_enc = CrtEnc()

    def test_init_keys(self):
        response = self.app.post('/init')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('public_keys', data)

    def test_process_ratings(self):
        test_ratings = [
            {'user_id': '1', 'item_id': '1', 'rating': 5.0}
        ]
        
        encrypted_ratings = []
        masks = []
        
        for r in test_ratings:
            mask_val = float(np.random.randint(0, 1 << 20))
            masked_rating = float(r['rating']) + mask_val
            
            ct = self.crt_enc.enc_crt(masked_rating)
            serialized_ct = [c.serialize().hex() for c in ct]
            
            encrypted_ratings.append({
                'user_id': r['user_id'],
                'item_id': r['item_id'],
                'ciphertext': serialized_ct
            })
            masks.append({'value': mask_val})

        response = self.app.post('/process_ratings', 
                               json={
                                   'encrypted_ratings': encrypted_ratings,
                                   'masks': masks
                               })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('processed_data', data)

    def test_fixed_point_ops(self):
        test_val = float(np.random.rand())
        encrypted_vec = self.crt_enc.enc_crt(test_val)
        serialized_vec = [ct.serialize().hex() for ct in encrypted_vec]
        
        response = self.app.post('/fixed_point_ops',
                               json={
                                   'vectors': {'data': serialized_vec},
                                   'operation': 'aggregate_users'
                               })
                               
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('result', data)

    def test_check_convergence(self):
        test_grads = {
            'user_gradients': (np.random.rand(20) * 0.001).tolist(),
            'item_gradients': (np.random.rand(20) * 0.001).tolist()
        }
        
        response = self.app.post('/check_conv',
                               json={
                                   'gradients': test_grads,
                                   'threshold': 0.01
                               })
                               
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('converged', data)

if __name__ == '__main__':
    unittest.main()
