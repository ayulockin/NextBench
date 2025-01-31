import weave


def get_dataset(ref: str) -> weave.Dataset:
    return weave.ref(ref).get()

DATASETS = {
    "MATH500": "weave:///ayut/NextBench/object/MATH500:YMovIwbHIlH2hxe70wriWldlxHwEfXnkfuDX4nvdbzw"
}

__all__ = [
    "get_dataset",
    "DATASETS",
]
