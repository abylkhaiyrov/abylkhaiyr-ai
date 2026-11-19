import sys
import time

from llm.core.config import DataConfig, load_yaml
from llm.data.pipeline import CorpusBuilder

started = time.monotonic()
manifest = CorpusBuilder(DataConfig.model_validate(load_yaml(sys.argv[1]))).build()
for name, info in manifest["sources"].items():
    print(name, info["seen"], info["kept"], info["rejected"], flush=True)
print(manifest["splits"], f"{time.monotonic() - started:.0f}s")
