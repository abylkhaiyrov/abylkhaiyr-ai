import sys

from llm.data.preference import build_preference_files, tokenize_preference_split
from pathlib import Path

from llm.tokenizer.bpe import BPETokenizer

root = Path(sys.argv[1])
tokenizer = BPETokenizer.load(sys.argv[2])
print(build_preference_files(root, max_pairs=int(sys.argv[3]), validation_pairs=100))
for split in ("train", "validation"):
    print(split, tokenize_preference_split(root, split, tokenizer))
