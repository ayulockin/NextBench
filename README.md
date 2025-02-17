# NextBench
NextBench is a collection of wide variety of benchmarks for accessing the performance of LLMs and VLMs and more.

This project aims to make it easy to run IMPORTANT benchmarks across multiple clients/sdks/providers. This is powered by [W&B Weave](https://weave-docs.wandb.ai).

If you would like to see a benchmark/client/sdk/provider added, please open an issue.

This is a work in progress but we will soon be opening it up for public contributions.


## Benchmarks

We have used the following benchmarks for NextBench. The list is not exhaustive and we will be adding more benchmarks in the future.

Each benchmark must have a `question` and `answer` column. If the original benchmark doesn't have the correct column names, we rename them as a post-processing step and upload the benchmark as a [W&B Weave Dataset](https://weave-docs.wandb.ai/guides/core-types/datasets). This allows us to consume the benchmarks in a consisten way and have better control over the versions of the benchmark especially for benchmarks that are updated frequently (MixEval).

- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa)
- [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite)

## Run Evaluation

An example evaluation command is shown below:

```bash
python eval.py --model-name gpt-4o-mini --scenario math500 --num-samples 100 --no-enable-cache
```

The following will load the MATH500 and MMLU-Pro datasets and run the evaluation using the exact match metric.

## TODOs

- [x] Scenarios are defined as classes in the `src/nextbench/scenarios` directory.
- [x] Caching of results.
- [x] Caching of datasets (because Weave caching is not working for datasets atm).
- [x] System prompts are defined as `weave.StringPrompt` and published to W&B for better tracking.
- [x] Configurable number of samples to evaluate from the dataset.
- [x] OpenAI client
- [x] Package everything as a CLI tool from eval.py file.
- [ ] DSPY prompt optimization.
- [ ] Add more clients (e.g. Groq, Anthropic, Gemini, etc.)
- [ ] Add more scenarios (e.g. GPQA-Diamond, LiveCodeBench, etc.)
- [x] Make it more configurable and user friendly.

<details>
<summary>Citations</summary>

```bibtex
@misc{wang2024mmluprorobustchallengingmultitask,
      title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark}, 
      author={Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
      year={2024},
      eprint={2406.01574},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.01574}, 
}
```

```bibtex
@misc{rein2023gpqagraduatelevelgoogleproofqa,
      title={GPQA: A Graduate-Level Google-Proof Q&A Benchmark}, 
      author={David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
      year={2023},
      eprint={2311.12022},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2311.12022}, 
}
```

```bibtex
@misc{jain2024livecodebenchholisticcontaminationfree,
      title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code}, 
      author={Naman Jain and King Han and Alex Gu and Wen-Ding Li and Fanjia Yan and Tianjun Zhang and Sida Wang and Armando Solar-Lezama and Koushik Sen and Ion Stoica},
      year={2024},
      eprint={2403.07974},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2403.07974}, 
}
```
