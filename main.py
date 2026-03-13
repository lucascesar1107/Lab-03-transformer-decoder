import numpy as np


# FUNÇÃO SOFTMAX

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# MÁSCARA CAUSAL (LOOK-AHEAD)


def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -np.inf, 0)
    return mask


def test_causal_mask():
    seq_len = 5

    Q = np.random.randn(seq_len, 4)
    K = np.random.randn(seq_len, 4)

    scores = Q @ K.T

    mask = create_causal_mask(seq_len)

    masked_scores = scores + mask

    probs = softmax(masked_scores)

    print("Scores originais:")
    print(scores)

    print("\nMáscara causal:")
    print(mask)

    print("\nProbabilidades após Softmax:")
    print(probs)


# CROSS ATTENTION

def cross_attention(encoder_out, decoder_state):

    d_model = encoder_out.shape[-1]

    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)

    Q = decoder_state @ Wq
    K = encoder_out @ Wk
    V = encoder_out @ Wv

    dk = K.shape[-1]

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dk)

    attention_weights = softmax(scores)

    output = attention_weights @ V

    return output


def test_cross_attention():

    encoder_output = np.random.randn(1, 10, 512)

    decoder_state = np.random.randn(1, 4, 512)

    result = cross_attention(encoder_output, decoder_state)

    print("\nSaída do Cross Attention:")
    print(result.shape)



# LOOP AUTO-REGRESSIVO

VOCAB_SIZE = 10000

VOCAB = ["token_" + str(i) for i in range(VOCAB_SIZE)]
VOCAB[0] = "<EOS>"


def generate_next_token(current_sequence, encoder_out):

    logits = np.random.randn(VOCAB_SIZE)

    probs = softmax(logits)

    return probs


def inference_loop():

    encoder_out = np.random.randn(1, 10, 512)

    sequence = ["<START>"]

    while True:

        probs = generate_next_token(sequence, encoder_out)

        next_token_id = np.argmax(probs)

        next_token = VOCAB[next_token_id]

        sequence.append(next_token)

        if next_token == "<EOS>":
            break

        if len(sequence) > 20:
            break

    print("\nFrase gerada:")
    print(sequence)



if __name__ == "__main__":

    print("==== TESTE MÁSCARA CAUSAL ====")
    test_causal_mask()

    print("\n==== TESTE CROSS ATTENTION ====")
    test_cross_attention()

    print("\n==== TESTE LOOP AUTOREGRESSIVO ====")
    inference_loop()
