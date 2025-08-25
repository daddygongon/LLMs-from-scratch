from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of someunknownPlace."
text = " <|endoftext|> ".join([text1, text2])
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
