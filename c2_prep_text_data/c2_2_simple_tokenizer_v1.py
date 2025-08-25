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

import re

# text = "Hello, world! This is a test."
preporocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preporocessed = [item for item in preporocessed if item.strip()]
print(preporocessed[:30])  # Print the first 30 tokens

all_tokens = sorted(set(preporocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print("Vocabulary size:", len(vocab))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(f"{i}: {item}")


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join([text1, text2])
print("Combined text:", text)
ids = tokenizer.encode(text)
print("Encoded IDs:", ids)
print("Decoded text:", tokenizer.decode(ids))
