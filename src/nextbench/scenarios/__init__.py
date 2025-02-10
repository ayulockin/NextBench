import weave
from .base_scenario import BaseScenario

DATASETS = {
    "MATH500": "weave:///ayut/NextBench/object/MATH500:YMovIwbHIlH2hxe70wriWldlxHwEfXnkfuDX4nvdbzw",
    "MMLU-Pro": "weave:///ayut/NextBench/object/MMLU-Pro:wYUCYcoPDOJgPFjEcKyqNNam0PI85uXA6LLuerXjnX4",
    "GPQA-Diamond": "weave:///ayut/NextBench/object/GPQA-Diamond:IMWhT1VCvVw53fSm0OqLtSrpAPJPPtOfvt2XzopRrGU",
    # "LiveCodeBench": "weave:///ayut/NextBench/object/LiveCodeBench:JCXfIIXRFx7gxUZEnCfEEv4k31epEsuTSu66G1MWAJs",
}

__all__ = [
    "BaseScenario",
    "DATASETS",
]
