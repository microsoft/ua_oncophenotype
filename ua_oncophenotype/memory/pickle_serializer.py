import base64
import pickle
import io


class RenameUnpickler(pickle.Unpickler):
    """Support backwards compatibility with old rwd_llm package name"""

    def find_class(self, module, name):
        if module.startswith("rwd_llm"):
            new_module = "ua_oncophenotype" + module[len("rwd_llm") :]
            print(f"changed module {module} -> {new_module}")
            module = new_module

        return super().find_class(module, name)


def rename_loads(data: bytes):
    """Load the pickle data (as bytes) with the custom unpickler."""
    # Wrap 'data' in a file-like object
    file_like = io.BytesIO(data)
    # Then use RenameUnpickler to load
    return RenameUnpickler(file_like).load()


class PickleSerializer:
    def __call__(self, obj):
        pickled_data = pickle.dumps(obj)
        base64_encoded = base64.b64encode(pickled_data)
        return base64_encoded.decode("utf-8")


class PickleDeserializer:
    def __call__(self, obj):
        base64_bytes = obj.encode("utf-8")
        pickled_data = base64.b64decode(base64_bytes)
        return rename_loads(pickled_data)
