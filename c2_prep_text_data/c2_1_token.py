import urllib.request

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])  # Print the first 100 character

import re

# text = "Hello, world! This is a test."
preporocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preporocessed = [item for item in preporocessed if item.strip()]
print(preporocessed[:30])  # Print the first 30 tokens

all_words = sorted(set(preporocessed))
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(f"{i:3d}: {item[0]:<10} -> {item[1]}")
    if i >= 50:
        break


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_|"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [seelf.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?_|"()\'])', r"\1", text)
        return text
