import numpy as np


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)


zhang_lecun_vocab = "abcdefghijklmnopqrstuvwxyz0123456789"


def to_one_hot(txt, vocab=zhang_lecun_vocab):
    vocab_hash = {b: a for a, b in enumerate(list(vocab))}
    vocab_size = len(vocab)
    one_hot_vec = np.zeros((1, vocab_size, len(txt)), dtype=np.float32)
    # run through txt and "switch on" relevant positions in one-hot vector
    for idx, char in enumerate(txt):
        try:
            vocab_idx = vocab_hash[char]
            one_hot_vec[0, vocab_idx, idx] = 1
        # raised if character is out of vocabulary
        except KeyError:
            pass
    return one_hot_vec


def get_one_hot_doc(txt, char_vocab, replace_vocab=None, replace_char=' ',
                    min_length=10, max_length=300, pad_out=True,
                    to_lower=True, reverse=True,
                    truncate_left=False, encoding=None):
    clean_txt = normalize(txt, replace_vocab, replace_char, min_length, max_length, pad_out,
                          to_lower, reverse, truncate_left, encoding)

    return to_one_hot(clean_txt, char_vocab)


def normalize(txt, vocab=None, replace_char=' ',
              min_length=10, max_length=300, pad_out=True,
              to_lower=True, reverse=True,
              truncate_left=False, encoding=None):
    # store length for multiple comparisons
    txt_len = len(txt)

    #     # normally reject txt if too short, but doing someplace else
    #     if txt_len < min_length:
    #         raise TextTooShortException("Too short: {}".format(txt_len))
    # truncate if too long
    if truncate_left:
        txt = txt[-max_length:]
    else:
        txt = txt[:max_length]
    # change case
    if to_lower:
        txt = txt.lower()
    # Reverse order
    if reverse:
        txt = txt[::-1]
    # replace chars
    if vocab is not None:
        txt = ''.join([c if c in vocab else replace_char for c in txt])
    # re-encode text
    if encoding is not None:
        txt = txt.encode(encoding, errors="ignore")
    # pad out if needed
    if pad_out and max_length > txt_len:
        txt = replace_char * (max_length - txt_len) + txt
    return txt


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if word not in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if word not in vocab:
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "bool":
        if word:
            return 1
        else:
            return 0
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word2vec[word] = vector
    if not silent:
        print("utils.py::create_vector => %s is missing" % word)
    return list(vector)
