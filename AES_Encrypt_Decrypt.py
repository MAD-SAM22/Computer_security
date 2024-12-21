import numpy as np
import hashlib
import os


S_box = np.array([0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16], dtype=np.uint8)

Inv_S_box = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
], dtype=np.uint8)

class AES:
    def __init__(self, password_str, salt):
        self.block_size = 16  # AES block size (128 bits)
        self.salt = salt

        # Encode the password and generate the key
        self.password = password_str.encode("UTF-8")
        self.key = self.KeyGeneration(self.password, self.salt)

        # Calculate key length in bytes
        self.key_len = self.key.shape[0] * 4  # Rows * 4 bytes per word
        if self.key_len not in [16, 24, 32]:  # Validate key length
            raise ValueError("Key length must be 128, 192, or 256 bits (16, 24, 32 bytes).")

        # Dynamically determine rounds based on key length
        self.rounds = {16: 10, 24: 12, 32: 14}[self.key_len]
        self.keys = self.KeyExpansion(self.key, self.rounds)



    def KeyGeneration(self, password, salt):
        """
        Dynamically generate a key for AES based on password length.
        Adjusts key length to match AES-compatible sizes (128, 192, 256 bits).
        """
        # AES-compatible key sizes
        valid_key_sizes = [16, 24, 32]  # 128, 192, 256 bits in bytes
        
        # Determine the closest valid key size (pad or truncate password if necessary)
        key_len = min(valid_key_sizes, key=lambda x: abs(len(password) - x))
        
        # Pad or truncate the password to match the key length
        padded_password = password.ljust(key_len, b"\x00")[:key_len]
        
        # Generate a key using scrypt
        key_bytes = hashlib.scrypt(
            padded_password, salt=salt, n=2**15, r=8, p=1, maxmem=2**26, dklen=key_len
        )

        # Ensure the key length is compatible for reshaping
        if len(key_bytes) % 4 != 0:
            raise ValueError("Key length must be a multiple of 4 bytes for reshaping.")

        # Reshape key for AES (4 columns per word)
        return np.frombuffer(key_bytes, dtype=np.uint8).reshape((-1, 4))



    def KeyExpansion(self, key, rounds):
        """
        Expands the input key into round keys for AES encryption.
        """
        N = key.shape[0]  # Number of rows in the key (4 bytes per row)
        R = rounds + 1  # Total number of round keys
        keys = np.zeros((4 * R, 4), dtype=np.uint8)  # Placeholder for expanded keys
        keys[:N] = key  # Copy the initial key into the first part of the key schedule

        # RCON initialization
        rcon = np.zeros((R, 4), dtype=np.uint8)
        rcon[0, 0] = 1  # Initial value for RCON

        for i in range(N, 4 * R):
            temp = keys[i - 1]
            if i % N == 0:
                temp = np.roll(temp, -1)  # Rotate word
                temp = np.vectorize(lambda x: S_box[x])(temp)  # SubBytes transformation
                temp[0] ^= rcon[i // N - 1, 0]  # XOR with RCON
            elif N > 6 and i % N == 4:
                temp = np.vectorize(lambda x: S_box[x])(temp)  # SubBytes for 256-bit keys
            keys[i] = keys[i - N] ^ temp  

        return keys.reshape(R, 4, 4).transpose(0, 2, 1)  # Transpose to match AES format



    def AddRoundKey(self, state, key):
        return np.bitwise_xor(state, key)

    def SubBytes(self, state):
        return np.vectorize(lambda x: S_box[x])(state)

    def ShiftRows(self, state):
        for i in range(1, 4):
            state[i] = np.roll(state[i], -i)
        return state

    def MixColumns(self, state):
        def mix_column(col):
            r = np.zeros_like(col)
            r[0] = (2 * col[0]) ^ (3 * col[1]) ^ col[2] ^ col[3]
            r[1] = col[0] ^ (2 * col[1]) ^ (3 * col[2]) ^ col[3]
            r[2] = col[0] ^ col[1] ^ (2 * col[2]) ^ (3 * col[3])
            r[3] = (3 * col[0]) ^ col[1] ^ col[2] ^ (2 * col[3])
            return r % 0x100
        return np.apply_along_axis(mix_column, 0, state)

    def encrypt_block(self, block):
        state = block.reshape(4, 4).copy()
        print("Initial State (Plaintext Block):")
        print(state)

        # AddRoundKey (Initial round)
        state = self.AddRoundKey(state, self.keys[0])
        print("\nAfter Initial AddRoundKey:")
        print(state)

        # Intermediate rounds
        for i, round_key in enumerate(self.keys[1:-1], start=1):
            print(f"\nRound {i}:")
            
            # SubBytes
            state = self.SubBytes(state)
            print("  After SubBytes:")
            print(state)
            
            # ShiftRows
            state = self.ShiftRows(state)
            print("  After ShiftRows:")
            print(state)
            
            # MixColumns
            state = self.MixColumns(state)
            print("  After MixColumns:")
            print(state)
            
            # AddRoundKey
            state = self.AddRoundKey(state, round_key)
            print("  After AddRoundKey:")
            print(state)

        # Final round (No MixColumns)
        print("\nFinal Round:")
        state = self.SubBytes(state)
        print("  After SubBytes:")
        print(state)
        
        state = self.ShiftRows(state)
        print("  After ShiftRows:")
        print(state)
        
        state = self.AddRoundKey(state, self.keys[-1])
        print("  After Final AddRoundKey:")
        print(state)

        return state.flatten()


    def InvSubBytes(self, state):
        return np.vectorize(lambda x: Inv_S_box[x])(state)

    def InvShiftRows(self, state):
        for i in range(1, 4):
            state[i] = np.roll(state[i], i)  # Shift rows to the right by the row index
        return state

    def InvMixColumns(self, state):
        def mix_column_inv(col):
            r = np.zeros_like(col)
            r[0] = (0x0e * col[0]) ^ (0x0b * col[1]) ^ (0x0d * col[2]) ^ (0x09 * col[3])
            r[1] = (0x09 * col[0]) ^ (0x0e * col[1]) ^ (0x0b * col[2]) ^ (0x0d * col[3])
            r[2] = (0x0d * col[0]) ^ (0x09 * col[1]) ^ (0x0e * col[2]) ^ (0x0b * col[3])
            r[3] = (0x0b * col[0]) ^ (0x0d * col[1]) ^ (0x09 * col[2]) ^ (0x0e * col[3])
            return r % 0x100
        return np.apply_along_axis(mix_column_inv, 0, state)

    def decrypt_block(self, block):
        state = block.reshape(4, 4).copy()
        print("Initial State (Ciphertext Block):")
        print(state)

        # Initial round key addition
        state = self.AddRoundKey(state, self.keys[-1])
        print("\nAfter Initial AddRoundKey:")
        print(state)

        # Intermediate rounds
        for i, round_key in enumerate(reversed(self.keys[1:-1]), start=1):
            print(f"\nRound {i}:")

            # Inverse ShiftRows
            state = self.InvShiftRows(state)
            print("  After InvShiftRows:")
            print(state)

            # Inverse SubBytes
            state = self.InvSubBytes(state)
            print("  After InvSubBytes:")
            print(state)

            # AddRoundKey
            state = self.AddRoundKey(state, round_key)
            print("  After AddRoundKey:")
            print(state)

            # Inverse MixColumns
            state = self.InvMixColumns(state)
            print("  After InvMixColumns:")
            print(state)

        # Final round (no MixColumns)
        print("\nFinal Round:")
        state = self.InvShiftRows(state)
        print("  After InvShiftRows:")
        print(state)

        state = self.InvSubBytes(state)
        print("  After InvSubBytes:")
        print(state)

        state = self.AddRoundKey(state, self.keys[0])
        print("  After Final AddRoundKey:")
        print(state)

        return state.flatten()


def xor_bytes(data1, data2):
    return bytes(a ^ b for a, b in zip(data1, data2))


def encrypt_decrypt(func, cipher, nonce, data, counter=0):
    result = b""
    for i in range(0, len(data), cipher.block_size):
        block = data[i:i + cipher.block_size]
        counter_bytes = counter.to_bytes(6, byteorder="big")
        IV = nonce + counter_bytes
        keystream = cipher.encrypt_block(np.frombuffer(IV, dtype=np.uint8))
        result += xor_bytes(keystream, block)
        counter += 1
    return result


def main():
    password = "very_long_password_example_with_padding`"
    salt = os.urandom(16)
    nonce = os.urandom(10)

    aes = AES(password, salt)

    plaintext = b"Example text to encrypt with AES. "
    padded_plaintext = plaintext.ljust((len(plaintext) // 16 + 1) * 16, b"\x00")

    print("\nPlaintext:", plaintext)

    # Encrypt and print steps
    print("\nEncryption Process:")
    ciphertext = encrypt_decrypt(aes.encrypt_block, aes, nonce, padded_plaintext)
    print("\nEncrypted Ciphertext:")
    print(ciphertext)

    decrypted_block = aes.decrypt_block(ciphertext)
    decrypted_plaintext = decrypted_block.tobytes().rstrip(b"\x00")
    print("\nDecrypted Plaintext:", decrypted_plaintext)



if __name__ == "__main__":
    main()
