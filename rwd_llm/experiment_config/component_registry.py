from typing import Any, Dict, List


class ComponentRegistry:
    def __init__(self, components: List[Dict[str, Any]]):
        """We use a list of dicts instead of a dict so that we can preserve order."""
        for pair in components:
            if len(pair) != 1:
                raise ValueError(f"Expected dict with 1 key, got {pair}")
            for name, component in pair.items():
                self.register(name, component)

    @classmethod
    def register(cls, name: str, component: Any):
        setattr(cls, name, component)

    @classmethod
    def get(cls, name: str) -> Any:
        return getattr(cls, name)
