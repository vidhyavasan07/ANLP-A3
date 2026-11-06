import json


class CharacterTokenizer:
    special_tokens = ("<pad>", "<bos>", "<sep>", "<eos>", "<unk>")

    def __init__(self, characters):
        self.characters = tuple(sorted(set(characters) - set(self.special_tokens)))
        self.tokens = self.special_tokens + self.characters
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = self.tokens
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.sep_id = self.token_to_id["<sep>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]

    @classmethod
    def from_texts(cls, texts):
        characters = set()
        for text in texts:
            characters.update(text)
        return cls(characters)

    @classmethod
    def from_jsonl(cls, cipher_path, plaintext_path):
        characters = set()
        for path, field in ((cipher_path, "ciphertext"), (plaintext_path, "text")):
            with open(path, encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    characters.update(row[field])
        return cls(characters)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text):
        return [self.token_to_id.get(character, self.unk_id) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        output = []
        special_ids = set(range(len(self.special_tokens)))
        for token_id in token_ids:
            index = int(token_id)
            if index < 0 or index >= len(self.id_to_token):
                token = "<unk>"
            else:
                token = self.id_to_token[index]
            if skip_special_tokens and index in special_ids:
                continue
            output.append(token)
        return "".join(output)

    def encode_pair(self, ciphertext, plaintext):
        return [self.bos_id, *self.encode(ciphertext), self.sep_id, *self.encode(plaintext), self.eos_id]

    def state_dict(self):
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_state_dict(cls, state):
        tokens = state["tokens"]
        if tuple(tokens[: len(cls.special_tokens)]) != cls.special_tokens:
            raise ValueError("Tokenizer state has an incompatible special-token order")
        return cls(tokens[len(cls.special_tokens) :])
