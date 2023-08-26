import hashlib
import os
from functools import lru_cache

from shared.config import UIConfig, load_config

CONFIGS_PATH = os.getenv("CONFIGS_PATH", "configs")


@lru_cache()
def get_configs():
    configs = {}
    for file in os.listdir(CONFIGS_PATH):
        if file.endswith(".yaml"):
            key = hashlib.md5(file.encode("utf-8")).hexdigest()
            configs[key] = UIConfig(
                config=load_config(os.path.join(CONFIGS_PATH, file)),
                file_name=file,
                key=key,
                file_path=os.path.join(CONFIGS_PATH, file),
            )
    return configs
