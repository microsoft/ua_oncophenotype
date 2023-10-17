import time
from concurrent.futures import ThreadPoolExecutor

from ...chains import NoOpChain
from ..kv_memory import KVMemory


def test_kv_memory_multithreading():
    """Without thread safety, the memories from different chain runs will overwrite each
    other."""
    n_threads = 4

    memory = KVMemory(keys=["a_out"])
    fake_chain_a = NoOpChain(
        expected_keys=["a"], the_key="a", output_key="a_out", memory=memory
    )
    fake_chain_b = NoOpChain(
        expected_keys=["a_out"], the_key="a_out", output_key="b_out", memory=memory
    )

    def run_prompt(inputs):
        fake_chain_a(inputs)
        time.sleep(0.1)
        # outputs from 'a' should be stored in memory
        return fake_chain_b({})

    inputs = [
        {"a": "a1"},
        {"a": "a2"},
        {"a": "a3"},
        {"a": "a4"},
    ]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(run_prompt, inputs))

    result_set = set(r["b_out"] for r in results)
    assert result_set == {"a1", "a2", "a3", "a4"}

    # check that keys from two memories in the same thread are kept separate
    m1 = KVMemory(keys=["a_val"])
    m2 = KVMemory(keys=["a_val"])
    m1.memories["a_val"] = "m1"
    m2.memories["a_val"] = "m2"
    assert m1.memories["a_val"] == "m1"
    assert m2.memories["a_val"] == "m2"


if __name__ == "__main__":
    test_kv_memory_multithreading()
