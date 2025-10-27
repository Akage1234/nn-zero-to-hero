from utility import get_stats, merge
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer():
    def __init__(self, pattern=None):
        self.vocab = None
        self.merges = None
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def register_special_tokens(self, special_token):
        self.special_tokens = special_token
        self.inverse_special_tokens = {v:k for k, v in special_token.items()}

    def train(self, text, vocab_size=256):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode('utf-8')) for ch in text_chunks]
        num_merges = vocab_size - 256
        merges = {}
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_id in ids:
                get_stats(chunk_id, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"merging {pair} to {idx}")
            ids = [merge(chunk_id, pair, idx) for chunk_id in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] +  vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids)>=2:
            pair = min(ids, key= lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids,pair, idx)
        return ids  

    def encode_simple(self, text):
        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:    
            chunk_utf = chunk.encode('utf-8')
            chunk_ids = self._encode_chunk(chunk_utf)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special='none_raise'):
        special = None
        if allowed_special=='all':
            special = self.special_tokens
        elif allowed_special=='none':
            special = {}
        elif allowed_special=='none_raise':
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k:v for k, v in self.special_tokens.items() if k in allowed_special}
        else: 
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special:
            return self.encode_simple(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        text_chunks = re.split(special_pattern, text)
        ids = []
        for chunk in text_chunks:    
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self.encode_simple(chunk))
        return ids

    def decode(self, ids):
        part_bytes = []
        for idx in ids: 
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid Token {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

# if __name__ == "__main__":
#     with open('tests/taylorswift.txt', 'r', encoding='utf-8') as f:
#         text = f.read()

#     tokenizer= RegexTokenizer()
#     tokenizer.train(text, vocab_size=100)
#     tokenizer.register_special_tokens({"<|endoftext|>": 100257})
#     ids = tokenizer.encode(text, allowed_special='all')
#     print(ids)
#     print(tokenizer.decode(ids))
#     print(tokenizer.decode(tokenizer.encode(text, allowed_special='all')) == text)
    
