from config import VOCAB_SIZE


class KTYTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode(
            "utf-8",
            errors="replace"
        )
