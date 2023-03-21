# 物联网001 孙宇晨 2206113828 仿射变换 加密解密

import string

# 定义仿射密码加密函数
def encrypt(text, a, b):
    # 定义字母表
    alphabet = string.ascii_uppercase

    # 将输入文本转换为大写字母
    text = text.upper()

    # 加密每个字符
    encrypted_text = ''
    for char in text:
        if char in alphabet:
            # 计算仿射变换后的字符
            char_index = alphabet.index(char)
            encrypted_char_index = (a * char_index + b) % 26
            encrypted_char = alphabet[encrypted_char_index]
            encrypted_text += encrypted_char
        else:
            # 如果字符不在字母表中，则将其保留为原始字符
            encrypted_text += char
    return encrypted_text


# 定义仿射密码解密函数
def decrypt(text, a, b):
    # 定义字母表
    alphabet = string.ascii_uppercase

    # 将输入文本转换为大写字母
    text = text.upper()

    # 计算a的逆元
    for i in range(26):
        if (a * i) % 26 == 1:
            a_inverse = i

    # 解密每个字符
    decrypted_text = ''
    for char in text:
        if char in alphabet:
            # 计算仿射变换后的字符
            char_index = alphabet.index(char)
            decrypted_char_index = (a_inverse * (char_index - b)) % 26
            decrypted_char = alphabet[decrypted_char_index]
            decrypted_text += decrypted_char
        else:
            # 如果字符不在字母表中，则将其保留为原始字符
            decrypted_text += char
    return decrypted_text

# 测试仿射密码加密和解密函数
text = 'HELLO WORLD'
a = 5
b = 7
encrypted_text = encrypt(text, a, b)
decrypted_text = decrypt(encrypted_text, a, b)
print('原始文本:', text)
print('加密后文本:', encrypted_text)
print('解密后文本:', decrypted_text)


text2 = input("Enter what you expected to enc\n")
encrypted_text2 = encrypt(text2, a, b)
decrypted_text2 = decrypt(encrypted_text2, a, b)
print('原始文本:', text2)
print('加密后文本:', encrypted_text2)
print('解密后文本:', decrypted_text2)
