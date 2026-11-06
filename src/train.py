import torch
from src.char_tokenizer import CharacterTokenizer
from src.transformer import DecoderOnlyTransformer
from functools import partial
from torch.utils.data import DataLoader
from src.sequence_dataset import PairedTextDataset, collate_sequences


tokenizer = CharacterTokenizer.from_jsonl(
    "data/train/cipher.jsonl",
    "data/train/cleaned.jsonl",
)

model = DecoderOnlyTransformer(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=512,
    pad_id=tokenizer.pad_id,
)


dataset = PairedTextDataset(
    "data/train/cipher.jsonl",
    "data/train/cleaned.jsonl",
    tokenizer,
    max_seq_len=512,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=partial(collate_sequences, pad_id=tokenizer.pad_id),
)

for input_ids, targets in loader:
    logits, loss = model(input_ids, targets)