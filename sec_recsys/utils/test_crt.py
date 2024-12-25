import math
from crt import CrtEnc

if __name__ == "__main__":
    # Initialize with 36-bit messages split into 3 parts
    crt_encryption = CrtEnc(bit_size=36, num_primes=3)
    
    # Original large number
    original = (1 << 35) + 42  # A 36-bit number
    
    # Encrypt using CRT optimization
    encrypted = crt_encryption.enc_crt(original)
    
    # Show ciphertext sizes
    print(f"Original value: {original}")
    print(f"Number of ciphertexts: {len(encrypted)}")
    for indx, ct in enumerate(encrypted):
        print(f"Cipher text {indx}: {ct.decrypt().tolist()[0]}\n")
    
    # Decrypt and verify
    decrypted = crt_encryption.dec_crt(encrypted)
    print(f"Decrypted value: {decrypted}")
    print(f"Decryption correct: {original == decrypted}")
