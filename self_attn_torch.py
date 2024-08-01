import torch

def self_attention(x, w_q, w_k, w_v, w_o):
    """
    Args: 
        x: bsz x seqlen x dim
        w_q: dim x nheads x head_dim
        w_k: dim x nheads x head_dim
        w_v: dim x nheads x head_dim
        w_o: nheads x head_dim x dim
    Returns:
        A: bsz x seqlen x dim
    """
    bsz, seqlen, dim = x.size()
    nheads, head_dim = w_q.size()[1], w_q.size()[2]

    # Linear transformations to get queries, keys, and values
    queries = x @ w_q.view(dim, -1)  # bsz x seqlen x (nheads * head_dim)
    keys = x @ w_k.view(dim, -1)    # bsz x seqlen x (nheads * head_dim)
    values = x @ w_v.view(dim, -1)  # bsz x seqlen x (nheads * head_dim)

    # Reshape queries, keys, and values to separate heads
    queries = queries.view(bsz, seqlen, nheads, head_dim).transpose(1, 2)  # bsz x nheads x seqlen x head_dim
    keys = keys.view(bsz, seqlen, nheads, head_dim).transpose(1, 2)      # bsz x nheads x seqlen x head_dim
    values = values.view(bsz, seqlen, nheads, head_dim).transpose(1, 2)  # bsz x nheads x seqlen x head_dim

    # Compute scaled dot-product attention
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (head_dim ** 0.5)  # bsz x nheads x seqlen x seqlen
    attention_weights = torch.softmax(attention_scores, dim=-1)  # bsz x nheads x seqlen x seqlen

    # Apply attention weights to values
    attention_output = torch.matmul(attention_weights, values)  # bsz x nheads x seqlen x head_dim
    attention_output = attention_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # bsz x seqlen x (nheads * head_dim)

    # Linear transformation to get the final output
    A = attention_output @ w_o.view(-1, dim)  # bsz x seqlen x dim

    return A

# Example usage
bsz, seqlen, dim = 5, 10, 8
nheads, head_dim = 2, 4

x = torch.randn(bsz, seqlen, dim)   
w_q = torch.randn(dim, nheads, head_dim)
w_k = torch.randn(dim, nheads, head_dim)
w_v = torch.randn(dim, nheads, head_dim)
w_o = torch.randn(nheads, head_dim, dim)

A = self_attention(x, w_q, w_k, w_v, w_o)
