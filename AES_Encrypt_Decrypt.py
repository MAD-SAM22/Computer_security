import os
import argparse
import sys
from tqdm import tqdm

def random_key_generator(key_length):
    return bytes.hex(os.urandom(key_length // 8))

class AES:
    Nb = 4
    Nk = 4
    Nr = 10

    Sbox = (
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    )

    InvSbox = (
        0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
        0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
        0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
        0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
        0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
        0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
        0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
        0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
        0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
        0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
        0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
        0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
        0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
        0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
        0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
    )

    Rcon = (
        0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
        0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
        0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
    )

    def __init__(self, key, mode=128):
        if mode == 192:
            self.Nk = 6
            self.Nr = 12
            self.key = self.text2matrix(key, 24)
        elif mode == 256:
            self.Nk = 8
            self.Nr = 14
            self.key = self.text2matrix(key, 32)
        else:
            self.key = self.text2matrix(key)

        self.key_expansion(self.key)

    def text2matrix(self, text, len=16):
        state = []
        for i in range(len):
            byte = int(text[i*2:i*2+2], 16)
            if i % 4 == 0:
                state.append([byte])
            else:
                state[i // 4].append(byte) 
        return state

    def matrix2text(self, s, len=16):
        text = ""
        for i in range(len // 4):
            for j in range(4):
                text += format(s[i][j], '02x')
        return text

    def sub_bytes(self, s):
        for i in range(self.Nb):
            for j in range(4):
                s[i][j] = self.Sbox[s[i][j]]
    
    def inv_sub_bytes(self, s):
        for i in range(self.Nb):
            for j in range(4):
                s[i][j] = self.InvSbox[s[i][j]]

    def shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]

    def inv_shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

    def xtime(self, b):
        if b & 0x80:
            b = b << 1
            b ^= 0x1B
        else:
            b = b << 1
        return b & 0xFF

    def mix_one_column(self, c):
        t = c[0] ^ c[1] ^ c[2] ^ c[3]
        u = c[0]
        c[0] ^= self.xtime(c[0] ^ c[1]) ^ t
        c[1] ^= self.xtime(c[1] ^ c[2]) ^ t
        c[2] ^= self.xtime(c[2] ^ c[3]) ^ t
        c[3] ^= self.xtime(c[3] ^ u) ^ t

    def mix_columns(self, s):
        for i in range(self.Nb):
            self.mix_one_column(s[i])

    def inv_mix_columns(self, s):
        for i in range(self.Nb):
            u = self.xtime(self.xtime(s[i][0] ^ s[i][2]))
            v = self.xtime(self.xtime(s[i][1] ^ s[i][3]))
            s[i][0] ^= u
            s[i][1] ^= v
            s[i][2] ^= u
            s[i][3] ^= v
        self.mix_columns(s)

    def add_round_key(self, s, k):
        for i in range(self.Nb):
            for j in range(4):
                s[i][j] ^= k[i][j]

    def sub_word(self, w):
        for i in range(len(w)):
            w[i] = self.Sbox[w[i]]

    def rotate_word(self, w):
        w[0], w[1], w[2], w[3] = w[1], w[2], w[3], w[0]

    def key_expansion(self, key):
        self.round_keys = self.key
        for i in range(self.Nk, self.Nb * (self.Nr + 1)):
            self.round_keys.append([0, 0, 0, 0])
            temp = self.round_keys[i - 1][:]
            if i % self.Nk == 0:
                self.rotate_word(temp)
                self.sub_word(temp)
                temp[0] = temp[0] ^ self.Rcon[i // self.Nk]
            elif self.Nk > 6 and i % self.Nk == 4:
                self.sub_word(temp)
            for j in range(4):
                self.round_keys[i][j] = self.round_keys[i - self.Nk][j] ^ temp[j]

    def cipher(self, text):
        self.state = self.text2matrix(text)
        self.add_round_key(self.state, self.round_keys[:4])
        for i in range(1, self.Nr):
            self.sub_bytes(self.state)
            self.shift_rows(self.state)
            self.mix_columns(self.state)
            self.add_round_key(self.state, self.round_keys[self.Nb * i : self.Nb * (i + 1)])
        self.sub_bytes(self.state)
        self.shift_rows(self.state)
        self.add_round_key(self.state, self.round_keys[len(self.round_keys) - 4:])
        return self.matrix2text(self.state)

    def decipher(self, text):
        self.encrypted_state = self.text2matrix(text)
        self.add_round_key(self.encrypted_state, self.round_keys[len(self.round_keys) - 4:])
        for i in range(self.Nr - 1, 0, -1):
            self.inv_shift_rows(self.encrypted_state)
            self.inv_sub_bytes(self.encrypted_state)
            self.add_round_key(self.encrypted_state, self.round_keys[self.Nb * i : self.Nb * (i + 1)])
            self.inv_mix_columns(self.encrypted_state)
        self.inv_shift_rows(self.encrypted_state)
        self.inv_sub_bytes(self.encrypted_state)
        self.add_round_key(self.encrypted_state, self.round_keys[:4])
        return self.matrix2text(self.encrypted_state)

def pad(block, block_length):
    bytes_to_pad = block_length - len(block) // 2
    for _ in range(bytes_to_pad):
        block += format(bytes_to_pad, '02x')
    return block

def unpad(block):
    bytes_to_unpad = int(block[-2:], 16)
    return block[:-bytes_to_unpad*2]

def xor_blocks(block_1, block_2):
    return format(int(block_1, 16) ^ int(block_2, 16), '032x')

def generate_random_iv(iv_length):
    return bytes.hex(os.urandom(iv_length))

def generate_random_ctr():
    return generate_random_iv(8) + "0000000000000000"

def increment_ctr(ctr):
    ctr_inc_int = int.from_bytes(bytes.fromhex(ctr), byteorder="big") + 1
    return bytes.hex(ctr_inc_int.to_bytes(length=16, byteorder="big"))

class ECB:
    def __init__(self, block_cipher_alg):
        self.block_cipher_alg = block_cipher_alg

    def cipher(self, filename, encrypted_file_name):
        hex_array = FileTools.open_file(filename, 32)
        if len(hex_array[-1]) < 32:
            hex_array[-1] = pad(hex_array[-1], 16)
        cipher_array = []
        for i in tqdm(range(len(hex_array)), desc="ECB encryption"):
            cipher_array.append(self.block_cipher_alg.cipher(hex_array[i]))
        FileTools.write_file(encrypted_file_name, cipher_array)

    def decipher(self, filename, decrypted_file_name):
        hex_array = FileTools.open_file(filename, 32)
        decrypted_array = []
        for i in tqdm(range(len(hex_array)), desc="ECB decryption"):
            decrypted_array.append(self.block_cipher_alg.decipher(hex_array[i]))
        decrypted_array[-1] = unpad(decrypted_array[-1])
        FileTools.write_file(decrypted_file_name, decrypted_array)

class CBC:
    def __init__(self, block_cipher_alg, iv_length):
        self.block_cipher_alg = block_cipher_alg
        self.iv = generate_random_iv(iv_length)

    def cipher(self, filename, encrypted_file_name):
        hex_array = FileTools.open_file(filename, 32)
        if len(hex_array[-1]) < 32:
            hex_array[-1] = pad(hex_array[-1], 16)
        cipher_array = [self.iv]
        iv = self.iv
        for i in tqdm(range(len(hex_array)), desc="CBC encryption"):
            block_to_cipher = xor_blocks(iv, hex_array[i])
            cipher_array.append(self.block_cipher_alg.cipher(block_to_cipher))
            iv = cipher_array[i + 1]
        FileTools.write_file(encrypted_file_name, cipher_array)

    def decipher(self, filename, decrypted_file_name):
        hex_array = FileTools.open_file(filename, 32)
        iv = hex_array[0]
        decrypted_array = []
        for i in tqdm(range(1, len(hex_array)), desc="CBC decryption"):
            decrypted_array.append(self.block_cipher_alg.decipher(hex_array[i]))
            decrypted_array[i - 1] = xor_blocks(iv, decrypted_array[i - 1])
            iv = hex_array[i]
        decrypted_array[-1] = unpad(decrypted_array[-1])
        FileTools.write_file(decrypted_file_name, decrypted_array)

class CTR:
    def __init__(self, block_cipher_alg):
        self.block_cipher_alg = block_cipher_alg
        self.ctr = generate_random_ctr()

    def cipher(self, filename, encrypted_file_name):
        hex_array = FileTools.open_file(filename, 32)
        cipher_array = [self.ctr]
        ctr = self.ctr
        for i in tqdm(range(len(hex_array)), desc="CTR encryption"):
            ctr_encrypted = self.block_cipher_alg.cipher(ctr)
            cipher_array.append(xor_blocks(ctr_encrypted, hex_array[i]))
            ctr = increment_ctr(ctr)
        FileTools.write_file(encrypted_file_name, cipher_array)

    def decipher(self, filename, decrypted_file_name):
        hex_array = FileTools.open_file(filename, 32)
        ctr = hex_array[0]
        decrypted_array = []
        for i in tqdm(range(1, len(hex_array)), desc="CTR decryption"):
            ctr_encrypted = self.block_cipher_alg.cipher(ctr)
            decrypted_array.append(xor_blocks(ctr_encrypted, hex_array[i]))
            ctr = increment_ctr(ctr)
        FileTools.write_file(decrypted_file_name, decrypted_array)

class FileTools:
    @staticmethod
    def open_file(filename, chunk_size):
        with open(filename, "rb") as f:
            hex_array = []
            for offset in range(0, os.path.getsize(filename), 16):
                hex_array.append(bytes.hex(f.read(16)))
                f.seek(offset + 16)
            f.close()
        return hex_array

    @staticmethod
    def write_file(filename, block_array):
        with open(filename, "ab") as f:
            for i in range(len(block_array)):
                f.write(bytes.fromhex(block_array[i]))
            f.close()

def main():
    # "encrypt" or "decrypt"
    operation = "decrypt"
    # cipher mode: "ECB", "CBC", or "CTR"
    mode = "CBC"
    # Define the input and output file paths
    input_file = "aes\encrypted_file.bin"
    output_file = "decrypted_file.bin"
    # Define the encryption/decryption key in hexadecimal
    # For AES-128, use a 32-character hex(128 bits)
    # For AES-192, use a 48-character hex(192 bits)
    # For AES-256, use a 64-character hex(256 bits)
    key = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    # key size in bits: 128, 192, or 256
    key_size = 128
    # -------------------------------Validation------------------------------------
    if key:
        key = key.lower()
        expected_length = key_size // 4
        if len(key) != expected_length:
            print(f"Error: Provided key length does not match the key size of {key_size} bits.")
            print(f"Expected key length: {expected_length} hex characters.")
            sys.exit(1)
        try:
            int(key, 16)
        except ValueError:
            print("Error: Key must be a valid hexadecimal string.")
            sys.exit(1)
    else:
        if operation == "encrypt":
            key = random_key_generator(key_size)
            print(f"Generated random key (hex): {key}")
        else:
            print("Error: Decryption requires a key. Please provide it in the script.")
            sys.exit(1)
    aes = AES(key, mode=key_size)
    if mode == "ECB":
        cipher_mode = ECB(aes)
    elif mode == "CBC":
        iv_length = 16
        cipher_mode = CBC(aes, iv_length=iv_length)
    elif mode == "CTR":
        cipher_mode = CTR(aes)
    else:
        print("Error: Unsupported cipher mode selected.")
        sys.exit(1)
    if operation == "encrypt":
        print(f"Starting encryption using {mode} mode...")
        cipher_mode.cipher(input_file, output_file)
        print(f"Encryption completed. Encrypted file saved as '{output_file}'.")
        if not key:
            print("Note: A random key was generated. Please save this key securely for decryption purposes.")
    elif operation == "decrypt":
        print(f"Starting decryption using {mode} mode...")
        cipher_mode.decipher(input_file, output_file)
        print(f"Decryption completed. Decrypted file saved as '{output_file}'.")
    else:
        print("Error: Unsupported operation. Choose 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
