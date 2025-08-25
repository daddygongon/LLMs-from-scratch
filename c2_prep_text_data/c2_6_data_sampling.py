from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4
for i in range(context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
