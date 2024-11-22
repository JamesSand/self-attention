import tensorflow as tf

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
    bsz, seqlen, dim = x.shape
    nheads, head_dim = w_q.shape[1], w_q.shape[2]

    # Linear transformations to get queries, keys, and values
    queries = tf.matmul(x, tf.reshape(w_q, (dim, -1)))  # bsz x seqlen x (nheads * head_dim)
    keys = tf.matmul(x, tf.reshape(w_k, (dim, -1)))    # bsz x seqlen x (nheads * head_dim)
    values = tf.matmul(x, tf.reshape(w_v, (dim, -1)))  # bsz x seqlen x (nheads * head_dim)

    # Reshape queries, keys, and values to separate heads
    queries = tf.reshape(queries, (bsz, seqlen, nheads, head_dim))
    queries = tf.transpose(queries, perm=[0, 2, 1, 3])  # bsz x nheads x seqlen x head_dim
    keys = tf.reshape(keys, (bsz, seqlen, nheads, head_dim))
    keys = tf.transpose(keys, perm=[0, 2, 1, 3])      # bsz x nheads x seqlen x head_dim
    values = tf.reshape(values, (bsz, seqlen, nheads, head_dim))
    values = tf.transpose(values, perm=[0, 2, 1, 3])  # bsz x nheads x seqlen x head_dim

    # Compute scaled dot-product attention
    attention_scores = tf.matmul(queries, tf.transpose(keys, perm=[0, 1, 3, 2])) / (head_dim ** 0.5)  # bsz x nheads x seqlen x seqlen
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # bsz x nheads x seqlen x seqlen

    # Apply attention weights to values
    attention_output = tf.matmul(attention_weights, values)  # bsz x nheads x seqlen x head_dim
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, (bsz, seqlen, -1))  # bsz x seqlen x (nheads * head_dim)

    # Linear transformation to get the final output
    A = tf.matmul(attention_output, tf.reshape(w_o, (-1, dim)))  # bsz x seqlen x dim

    return A

# Example usage
bsz, seqlen, dim = 5, 10, 8
nheads, head_dim = 2, 4

x = tf.random.normal((bsz, seqlen, dim))   
w_q = tf.random.normal((dim, nheads, head_dim))
w_k = tf.random.normal((dim, nheads, head_dim))
w_v = tf.random.normal((dim, nheads, head_dim))
w_o = tf.random.normal((nheads, head_dim, dim))

A = self_attention(x, w_q, w_k, w_v, w_o)
