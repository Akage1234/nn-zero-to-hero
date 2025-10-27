from utility import get_stats, merge

class BasicTokenizer():
    def __init__(self):
        self.vocab = None
        self.merges = None

    def train(self, text, vocab_size=256):
        # text = text[:1000]
        ids = list(text.encode('utf-8'))
        n_chrs = len(set(ids))
        num_merges = vocab_size - n_chrs
        merges = {}
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"merging {pair} to {idx}")
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] +  vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        while len(ids)>=2:
            pair = min(ids, key= lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids,pair, idx)
        return ids    

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

# if __name__ == "__main__":
#     with open('tests/taylorswift.txt', 'r', encoding='utf-8') as f:
#         text = f.read()

#     tokenizer= BasicTokenizer()
#     tokenizer.train(text, vocab_size=100)
#     ids = tokenizer.encode(text)
#     # print(ids)
#     # print(tokenizer.decode(ids))
#     print(tokenizer.decode(tokenizer.encode(text)) == text)
    
