import numpy as np
import tenseal as ts
from typing import List, Tuple
from sympy import primefactors, mod_inverse
import math

class CrtEnc:
    '''
    Implement of Chinese Remainder Theorem on CKKS for large message encryption
    '''
    def __init__(self, bit_size=36, num_primes=3):
        '''
        bit_size: bits for msg space
        num_primes: number of prime num to use
        '''
        self.bit_size = bit_size
        self.num_primes = num_primes
        self.prime_bit_size = math.ceil(bit_size/num_primes)
        self.primes = self._gen_primes()
        self.contexts = self._init_contexts()

    def _gen_primes(self):
        '''
        return list[int]
        '''
        lprimes = []
        current = (1 << self.prime_bit_size) -1
        
        while len(lprimes) < self.num_primes:
            while current > 0:
                if self._is_prime(current):
                    lprimes.append(current)
                    break
                current -= 2
            current -= 2

        return lprimes
   
    def _init_contexts(self):
        '''
        Initialize context of CKKS for each prime
        '''
        contexts = []
        for _ in self.primes:
            context = ts.Context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[40, 20, 20, 40]
                    )
            context.generate_galois_keys()
            context.global_scale = 2**20
            contexts.append(context)
        return contexts
    
    def enc_crt(self, value):
        '''
        Return list of ciphertext, one per prime
        '''
        residues = [value % p for p in self.primes]
        cts = []
        for i, residue in enumerate(residues):
            encrypted = ts.ckks_tensor(self.contexts[i],[float(residue)])
            cts.append(encrypted)

        return cts
    
    def dec_crt(self, cts):
        residues = []
        for i, ct in enumerate(cts):
            decrypted = ct.decrypt()
            residue = int(round(decrypted.tolist()[0]))
            residues.append(residue)
        return self._reconstruct_crt(residues)
    
    def _is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def add_enc(self, ct1, ct2):
        '''
        Addition for two CRT-enc ciphertext
        '''
        result = []
        for c1, c2 in zip(ct1, ct2):
            result.append(c1 + c2)
        return result

    def mul_enc(self, ct1, ct2):
        '''
        Multiplication for two CRT-enc ciphertext
        '''
        result = []
        for c1, c2 in zip(ct1, ct2):
            result.append(c1 * c2)
        return result


    def _reconstruct_crt(self, residues):
        '''
        Reconstruct origin number using crt
        N = Sum(r_i * m_i * m_i^-1) mod M
        ---
        M = product all primes
        m_i = M/prime_i
        m_i^-1 = modular inverse of m_i mod prime_i
        '''
        total = 0
        product = math.prod(self.primes)
        for residue, prime in zip(residues, self.primes):
            p = product // prime
            total += residue * mod_inverse(p, prime) * p
        return total % product


