import tenseal as ts
from typing import Dict, Any
import json

class KeyManager:
    def __init__(self):
        self.context = None
        self.pub_key = None
        self.sec_key = None

    def gen_keys(self):
        self.context = ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[40, 20, 20, 40]
            )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**20
        self.pub_key = self.context.serialize(save_secret_key=False)
        self.sec_key = self.context.secret_key()
        return {
            "pub_key": self.pub_key.hex(),
            "galois_keys": self.context.galois_keys().hex()
        }

    def decrypt(self, enc_data):
        if not self.context:
            raise ValueError("Not initialize keys")
        vector = ts.ckks_vector_from(self.context, enc_data)
        return vec.decrypt().tolist()[0]

    def save_keys(self, path):
        if not self.context:
            raise ValueError("Not initialize keys")

        with open(path, 'w') as f:
            json.dump({
                    "pub_key": self.pub_key.hex(),
                    "sec_key": self.sec_key.hex()
                    }, f)
