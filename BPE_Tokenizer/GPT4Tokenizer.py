import tiktoken
from .RegexTokenizer import RegexTokenizer

## Utility functions to recover merges
def bpe(mergeable_ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        ix0, ix1 = mergeable_ranks[pair[0]], mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

## lightweight wrapper on Regex
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding('cl100k_base')
        mergeable_ranks = enc._mergeable_ranks
        # print(mergeable_ranks)
        self.merges = recover_merges(mergeable_ranks)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        self.bytes_shuffle = {i:mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_bytes_shuffle = {v:k for k, v in self.bytes_shuffle.items()}
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        text_bytes = bytes(self.bytes_shuffle[b] for b in text_bytes)
        return super()._encode_chunk(text_bytes)
    
    def decode(self, ids):
        segments = []
        for idx in ids:
            if idx in self.vocab:
                segments.append((True, self.vocab[idx]))  
            elif idx in self.inverse_special_tokens:
                segments.append((False, self.inverse_special_tokens[idx].encode("utf-8")))  # raw bytes
            else:
                raise ValueError(f"Invalid Token {idx}")

        out = bytearray()
        for need_unshuffle, seg in segments:
            if need_unshuffle:
                out.extend(self.inverse_bytes_shuffle[b] for b in seg)
            else:
                out.extend(seg)
        return bytes(out).decode('utf-8', errors='replace')
        

if __name__ == "__main__":
    with open('tests/taylorswift.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer= GPT4Tokenizer()
    tokenizer.train(text, vocab_size=100)
    ids = tokenizer.encode(text, allowed_special='all')
    print(ids)
    print(tokenizer.decode(ids))
    print(tokenizer.decode(tokenizer.encode(text, allowed_special='all')) == text)
    
