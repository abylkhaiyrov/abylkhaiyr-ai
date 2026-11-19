import json
import sys

from llm.data.sft_builder import build_sft_groups
from llm.tokenizer.bpe import BPETokenizer

tokenizer = BPETokenizer.load(sys.argv[2])
counts = build_sft_groups(sys.argv[1], tokenizer, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
print(json.dumps(counts))
