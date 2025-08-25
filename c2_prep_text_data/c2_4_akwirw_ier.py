from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")

text = "Aikirw ier"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
