import tenseal as ts
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Random import get_random_bytes
from typing import Dict, Any
import json
import base64

class KeyManager:
    def __init__(self):
        self.ahe_key = None
        self.he_context = None

    def gen_keys(self):
        self.ahe_key = ElGamal.generate(bits=1024, randfunc=get_random_bytes)
        
        pub_key = self.ahe_key.publickey()

        key_data = (
            self.int_to_bytes(pub_key._keydata['p'], 128) +
            self.int_to_bytes(pub_key._keydata['g'], 128) +
            self.int_to_bytes(pub_key._keydata['y'], 128)
                )

        self.he_context= ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[40, 20, 20, 40]
            )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**20
        
         
        return {
            "ahe_pkey": base64.b64encode(key_data).decode(),
            "he_context": self.he_context.serialize(save_galois_keys=True).hex()
        }

    def decrypt_ahe(self, ciphertext):
        if not self.ahe_key:
            raise ValueError("Not initialize AHE key")
        enc_data = base64.b64decode(ciphertext)
        return self.ahe_key.decrypt(enc_data)

    
    def int_to_bytes(self, x, length):
        return int(x).to_bytes(length, byteorder='big')
