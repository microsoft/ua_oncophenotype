import base64
import pickle


class PickleSerializer:
    def __call__(self, obj):
        pickled_data = pickle.dumps(obj)
        base64_encoded = base64.b64encode(pickled_data)
        return base64_encoded.decode("utf-8")


class PickleDeserializer:
    def __call__(self, obj):
        base64_bytes = obj.encode("utf-8")
        pickled_data = base64.b64decode(base64_bytes)
        return pickle.loads(pickled_data)
