## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

We also provide a number of optional dependencies for extended functionality. Extras can be installed via `pip install -e ".[NAME]"`

| Name          | Use                                   |
|---------------|---------------------------------------|
| anthropic     | For using Anthropic's models          |
| dev           | For linting PRs and contributions     |
| gptq          | For loading models with GPTQ          |
| ifeval        | For running the IFEval task           |
| mamba         | For loading Mamba SSM models          |
| math          | For running math task answer checking |
| multilingual  | For multilingual tokenizers           |
| openai        | For using OpenAI's models             |
| promptsource  | For using PromptSource prompts        |
| sentencepiece | For using the sentencepiece tokenizer |
| testing       | For running library test suite        |
| vllm          | For loading models with vLLM          |
| zeno          | For visualizing results with Zeno     |
|---------------|---------------------------------------|
| all           | Loads all extras (not recommended)    |


### Tensor + Data Parallel and Optimized Inference with `vLLM`

We also support vLLM for faster inference on [supported model types](https://docs.vllm.ai/en/latest/models/supported_models.html). For single-GPU or multi-GPU — tensor parallel, data parallel, or a combination of both — inference, for example:

```bash
lm_eval --model vllm \
    --model_args pretrained={model_name},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
    --tasks lambada_openai \
    --batch_size auto
```
For a full list of supported vLLM configurations, please reference our vLLM integration and the vLLM documentation.

vLLM occasionally differs in output from Huggingface. We treat Huggingface as the reference implementation, and provide a [script](./scripts/model_comparator.py) for checking the validity of vllm results against HF.

### Model APIs and Inference Servers

Our library also supports the evaluation of models served via several commercial APIs, and we hope to implement support for the most commonly used performant local/self-hosted inference servers.

To call a hosted model, use:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval --model openai-completions \
    --model_args model=davinci \
    --tasks lambada_openai,hellaswag
```

We also support using your own local inference server with an implemented version of the OpenAI ChatCompletions endpoint and passing trained HuggingFace artifacts and tokenizers.

```bash
lm_eval --model local-chat-completions --tasks gsm8k --model_args model=facebook/opt-125m,base_url=http://{yourip}:8000/v1
```
Note that for externally hosted models, configs such as `--device` and `--batch_size` should not be used and do not function. Just like you can use `--model_args` to pass arbitrary arguments to the model constructor for local models, you can use it to pass arbitrary arguments to the model API for hosted models. See the documentation of the hosting service for information on what arguments they support.

| API or Inference Server                                                                                                   | Implemented?                    | `--model <xxx>` name                                                | Models supported:                                                                             | Request Types:                                             |
|---------------------------------------------------------------------------------------------------------------------------|---------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------|
| OpenAI Completions                                                                                                        | :heavy_check_mark:              | `openai-completions` | up to `code-davinci-002`                                                                      | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| OpenAI ChatCompletions                                                                                                    | :heavy_check_mark:        | `openai-chat-completions`, `local-chat-completions`                                                               | [All ChatCompletions API models](https://platform.openai.com/docs/guides/gpt)                 | `generate_until` (no logprobs)                             |
| Anthropic                                                                                                                 | :heavy_check_mark:              | `anthropic`                                                         | [Supported Anthropic Engines](https://docs.anthropic.com/claude/reference/selecting-a-model)  | `generate_until` (no logprobs)                             |
| Textsynth                                                                                                                 | :heavy_check_mark:                   | `textsynth`                                                         | [All supported engines](https://textsynth.com/documentation.html#engines)                     | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Cohere                                                                                                                    | [:hourglass: - blocked on Cohere API bug](https://github.com/EleutherAI/lm-evaluation-harness/pull/395) | N/A                                                                 | [All `cohere.generate()` engines](https://docs.cohere.com/docs/models)                        | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) (via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)) | :heavy_check_mark:              | `gguf`, `ggml`                                                      | [All models supported by llama.cpp](https://github.com/ggerganov/llama.cpp)                   | `generate_until`, `loglikelihood`, (perplexity evaluation not yet implemented) |
| vLLM                                                                                                                      | :heavy_check_mark:       | `vllm`                                                              | [Most HF Causal Language Models](https://docs.vllm.ai/en/latest/models/supported_models.html) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Mamba                       | :heavy_check_mark:       | `mamba_ssm`                                                                      | [Mamba architecture Language Models via the `mamba_ssm` package](https://huggingface.co/state-spaces) | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                             |
| Your local inference server!                                                                                              | :heavy_check_mark:                             | `local-chat-completions` (using `openai-chat-completions` model type)    | Any server address that accepts GET requests using HF models and mirror's OpenAI's ChatCompletions interface                                  | `generate_until`                                           |                                | ...                                                      |

It is on our roadmap to create task variants designed to enable models which do not serve logprobs/loglikelihoods to be compared with generation performance of open-source models.

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

### Additional Features

If you have a Metal compatible Mac, you can run the eval harness using the MPS back-end by replacing `--device cuda:0` with `--device mps` (requires PyTorch version 2.1 or higher).

> [!Note]
> You can inspect what the LM inputs look like by running the following command:
> ```bash
> python write_out.py \
>     --tasks all_tasks \
>     --num_fewshot 5 \
>     --num_examples 10 \
>     --output_base_path /path/to/output/folder
> ```
> This will write out one text file for each task.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
lm_eval --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

## Advanced Usage Tips

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

[GPTQ](https://github.com/PanQiWei/AutoGPTQ) quantized models can be loaded by specifying their file names in `,gptq=NAME` (or `,gptq=True` for default names) in the `model_args` argument:

```bash
lm_eval --model hf \
    --model_args pretrained=model-name-or-path,gptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

To save evaluation results provide an `--output_path`. We also support logging model responses with the `--log_samples` flag for post-hoc analysis.

Additionally, one can provide a directory with `--use_cache` to cache the results of prior runs. This allows you to avoid repeated execution of the same (model, task) pairs for re-scoring.

For a full list of supported arguments, check out the [interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) guide in our documentation!

## Visualizing Results

You can use [Zeno](https://zenoml.com) to visualize the results of your eval harness runs.

First, head to [hub.zenoml.com](https://hub.zenoml.com) to create an account and get an API key [on your account page](https://hub.zenoml.com/account).
Add this key as an environment variable:

```bash
export ZENO_API_KEY=[your api key]
```

You'll also need to install the `lm_eval[zeno]` package extra.

To visualize the results, run the eval harness with the `log_samples` and `output_path` flags.
We expect `output_path` to contain multiple folders that represent individual model names.
You can thus run your evaluation on any number of tasks and models and upload all of the results as projects on Zeno.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path output/gpt-j-6B
```

Then, you can upload the resulting data using the `zeno_visualize` script:

```bash
python scripts/zeno_visualize.py \
    --data_path output \
    --project_name "Eleuther Project"
```

This will use all subfolders in `data_path` as different models and upload all tasks within these model folders to Zeno.
If you run the eval harness on multiple tasks, the `project_name` will be used as a prefix and one project will be created per task.

You can find an example of this workflow in [examples/visualize-zeno.ipynb](examples/visualize-zeno.ipynb).

