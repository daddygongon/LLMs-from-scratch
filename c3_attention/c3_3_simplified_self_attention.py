import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum", attn_weights_2_tmp.sum())


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum()


attn_weights_2_tmp = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_tmp)
print("Sum", attn_weights_2_tmp.sum())

attn_weights_2_tmp = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2_tmp)
print("Sum", attn_weights_2_tmp.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2_tmp[i] * x_i

print("Context vector:", context_vec_2)

attn_scores = inputs @ inputs.T
print("Attention scores:\n", attn_scores)
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention weights:\n", attn_weights)
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print("All context vectors:\n", all_context_vecs)
