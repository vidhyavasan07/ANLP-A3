"""Build decoder-only training sequences from paired ciphertext and text JSONL files."""

import json

import torch
from torch.utils.data import Dataset


def build_input_target(ciphertext, plaintext, tokenizer, max_seq_len):
    """Encode one pair for next-token training.

    The model receives ``<bos> ciphertext <sep> plaintext`` and learns to
    predict only the plaintext and final ``<eos>``.  The ciphertext is prompt
    context, so its next-token labels are set to ``pad_id`` and ignored by the
    transformer's loss function.

    ``max_seq_len`` is the number of input positions.  Long pairs are clipped
    while preserving both ``<sep>`` and ``<eos>``; at least one plaintext
    character is retained when the limit permits it.
    """
    if max_seq_len < 4:
        raise ValueError("max_seq_len must be at least 4")

    cipher_ids = tokenizer.encode(ciphertext)
    text_ids = tokenizer.encode(plaintext)

    # Reserve space for BOS, SEP, EOS, and at least one target character.
    # Keep the source prompt to roughly half the context for long examples.
    cipher_limit = min(len(cipher_ids), max(1, (max_seq_len - 2) // 2))
    cipher_ids = cipher_ids[:cipher_limit]
    text_limit = max_seq_len - 2 - len(cipher_ids)
    text_ids = text_ids[:text_limit]

    full_ids = [tokenizer.bos_id, *cipher_ids, tokenizer.sep_id, *text_ids, tokenizer.eos_id]
    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
    targets = torch.tensor(full_ids[1:], dtype=torch.long)

    # The target at the SEP position is the first plaintext token, so only
    # positions before SEP are masked from the language-model loss.
    sep_position = len(cipher_ids) + 1
    targets[:sep_position] = tokenizer.pad_id
    return input_ids, targets


class PairedTextDataset(Dataset):
    """Read aligned ciphertext/plaintext JSONL files without loading them all."""

    def __init__(self, cipher_path, plaintext_path, tokenizer, max_seq_len):
        self.cipher_path = str(cipher_path)
        self.plaintext_path = str(plaintext_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._offsets = []

        # Keep only byte offsets in memory; the training corpus itself can be
        # large. Validate pair IDs while building the index.
        with open(self.cipher_path, "rb") as cipher_file, open(
            self.plaintext_path, "rb"
        ) as plaintext_file:
            while True:
                cipher_offset = cipher_file.tell()
                plaintext_offset = plaintext_file.tell()
                cipher_line = cipher_file.readline()
                plaintext_line = plaintext_file.readline()
                if not cipher_line and not plaintext_line:
                    break
                if not cipher_line or not plaintext_line:
                    raise ValueError("Ciphertext and plaintext files have different row counts")
                cipher_row = json.loads(cipher_line)
                plaintext_row = json.loads(plaintext_line)
                if cipher_row.get("id") != plaintext_row.get("id"):
                    raise ValueError(
                        f"Mismatched pair IDs: {cipher_row.get('id')!r} and "
                        f"{plaintext_row.get('id')!r}"
                    )
                self._offsets.append((cipher_offset, plaintext_offset))

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index):
        cipher_offset, plaintext_offset = self._offsets[index]
        with open(self.cipher_path, "rb") as cipher_file, open(
            self.plaintext_path, "rb"
        ) as plaintext_file:
            cipher_file.seek(cipher_offset)
            plaintext_file.seek(plaintext_offset)
            cipher_row = json.loads(cipher_file.readline())
            plaintext_row = json.loads(plaintext_file.readline())

        return build_input_target(
            cipher_row["ciphertext"],
            plaintext_row["text"],
            self.tokenizer,
            self.max_seq_len,
        )


def collate_sequences(batch, pad_id):
    """Pad variable-length examples into ``(input_ids, targets)`` tensors."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_length = max(input_ids.numel() for input_ids, _ in batch)
    input_batch = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    target_batch = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for row, (input_ids, targets) in enumerate(batch):
        length = input_ids.numel()
        input_batch[row, :length] = input_ids
        target_batch[row, :length] = targets
    return input_batch, target_batch
