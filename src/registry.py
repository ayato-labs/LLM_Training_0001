from typing import Dict, Any, Callable

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Any] = {}

    def register(self, name: str):
        def decorator(cls):
            self._entries[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._entries:
            raise ValueError(f"{name} is not registered in {self.name}")
        return self._entries[name]

# コンポーネントごとのレジストリ
MODEL_REGISTRY = Registry("Models")
OPTIMIZER_REGISTRY = Registry("Optimizers")
DATASET_REGISTRY = Registry("Datasets")
