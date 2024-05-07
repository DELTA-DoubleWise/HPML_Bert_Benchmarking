# Transformers

We submitted another Github Repo, which is forked and modified based on the official HuggingFace Transformer Library. Therefore, we won't modify the README in that directory as the usecase should be exactly the same as the original Transformers. We noted the files that we made changes to here for your reference:

    -  src/transformers/models/align/modeling_align.py
    -  src/transformers/models/altclip/modeling_altclip.py
    -  src/transformers/models/bert/modeling_bert.py
    -  src/transformers/models/bert_generation/modeling_bert_generation.py
    -  src/transformers/models/bridgetower/modeling_bridgetower.py
    -  src/transformers/models/camembert/modeling_camembert.py
    -  tests/models/bert/test_modeling_bert.py


# HPML_Bert_Benchmarking

This repository contains scripts for benchmarking BERT models across different tasks such as masked language modeling (MLM), sequence classification, token classification, and question answering. The scripts utilize the Hugging Face Transformers library (which we added the support for FlashAttention2) to load pre-trained models and perform benchmarks on specified tasks with options for various configurations.

## Code outline

    HPML_BERT_BENCHMARKING/
    ├── classification/                    # Classification task results
    │   ├── combined_plot.png              # Combined plot for classification task
    │   ├── log_Bert_flash_attention_2_classification.csv  # Benchmark results for classification task using FlashAttention2
    │   ├── log_Bert_sdpa_classification.csv              # Benchmark results for classification task using SDPA
    │   ├── SeqLen_128_Mem_saved_(%).png   # Memory saved plot for classification task
    │   ├── SeqLen_128_Speedup_(%).png     # Speedup plot for classification task
    │   ├── SeqLen_256_Mem_saved_(%).png   # Memory saved plot for classification task
    │   ├── SeqLen_256_Speedup_(%).png     # Speedup plot for classification task
    │   ├── SeqLen_512_Mem_saved_(%).png   # Memory saved plot for classification task
    │   └── SeqLen_512_Speedup_(%).png     # Speedup plot for classification task
    │
    ├── mlm/                               # Masked Language Modeling task results
    │   ├── combined_plot.png              # Combined plot for MLM task
    │   ├── log_Bert_flash_attention_2_mlm.csv            # Benchmark results for MLM task using FlashAttention2
    │   ├── log_Bert_sdpa_mlm.csv                         # Benchmark results for MLM task using SDPA
    │   ├── SeqLen_128_Mem_saved_(%).png   # Memory saved plot for MLM task
    │   ├── SeqLen_128_Speedup_(%).png     # Speedup plot for MLM task
    │   ├── SeqLen_256_Mem_saved_(%).png   # Memory saved plot for MLM task
    │   ├── SeqLen_256_Speedup_(%).png     # Speedup plot for MLM task
    │   ├── SeqLen_512_Mem_saved_(%).png   # Memory saved plot for MLM task
    │   └── SeqLen_512_Speedup_(%).png     # Speedup plot for MLM task
    │
    ├── qa/                                # Question Answering task results
    │   ├── combined_plot.png              # Combined plot for QA task
    │   ├── log_Bert_flash_attention_2_qa.csv             # Benchmark results for QA task using FlashAttention2
    │   ├── log_Bert_sdpa_qa.csv                          # Benchmark results for QA task using SDPA
    │   ├── SeqLen_128_Mem_saved_(%).png   # Memory saved plot for QA task
    │   ├── SeqLen_128_Speedup_(%).png     # Speedup plot for QA task
    │   ├── SeqLen_256_Mem_saved_(%).png   # Memory saved plot for QA task
    │   ├── SeqLen_256_Speedup_(%).png     # Speedup plot for QA task
    │   ├── SeqLen_512_Mem_saved_(%).png   # Memory saved plot for QA task
    │   └── SeqLen_512_Speedup_(%).png     # Speedup plot for QA task
    │
    ├── token_classification/              # Token Classification task results
    │   ├── combined_plot.png              # Combined plot for token classification task
    │   ├── log_Bert_flash_attention_2_token_classification.csv  # Benchmark results for token classification task using FlashAttention2
    │   ├── log_Bert_sdpa_token_classification.csv        # Benchmark results for token classification task using SDPA
    │   ├── SeqLen_128_Mem_saved_(%).png   # Memory saved plot for token classification task
    │   ├── SeqLen_128_Speedup_(%).png     # Speedup plot for token classification task
    │   ├── SeqLen_256_Mem_saved_(%).png   # Memory saved plot for token classification task
    │   ├── SeqLen_256_Speedup_(%).png     # Speedup plot for token classification task
    │   ├── SeqLen_512_Mem_saved_(%).png   # Memory saved plot for token classification task
    │   └── SeqLen_512_Speedup_(%).png     # Speedup plot for token classification task
    │
    ├── results/                           # Benchmark results
    │   ├── log_Bert_flash_attention_2_classification.csv  # Benchmark results for classification task using FlashAttention2
    │   ├── log_Bert_flash_attention_2_mlm.csv            # Benchmark results for MLM task using FlashAttention2
    │   ├── log_Bert_flash_attention_2_qa.csv             # Benchmark results for QA task using FlashAttention2
    │   ├── log_Bert_flash_attention_2_token_classification.csv  # Benchmark results for token classification task using FlashAttention2
    │   ├── log_Bert_sdpa_classification.csv              # Benchmark results for classification task using SDPA
    │   ├── log_Bert_sdpa_mlm.csv                        # Benchmark results for MLM task using SDPA
    │   ├── log_Bert_sdpa_qa.csv                         # Benchmark results for QA task using SDPA
    │   └── log_Bert_sdpa_token_classification.csv        # Benchmark results for token classification task using SDPA
    │
    ├── benchmark_Bert.py                  # Python script for benchmarking BERT models
    ├── plot.py                            # Python script for plotting benchmark results using csv files in the results directory
    ├── run_Bert.sh                        # Shell script for running the benchmarking script
    ├── .gitignore                         # Specifies intentionally untracked files to ignore
    ├── LICENSE                            # The LICENSE file
    └── README.md                          # The top-level README for developers using this project



## Installation

Follow these steps to set up the environment and install the required packages:

1. **Clone the Custom Transformers Repository**
   ```bash
   git clone https://github.com/DELTA-DoubleWise/transformers.git
   ```

2. **Set Up a Virtual Environment** (optional but recommended)
   Navigate to the benchmarking directory where you want to run benchmarks and create a virtual environment:

3. **Install the Custom Transformers Library**
   Change to the cloned `transformers` directory and install the library in editable mode:
   ```bash
   cd transformers
   pip install -e .
   ```

4. **Install Other Required Packages**
   After installing the custom transformers library, install other required packages:
   ```bash
   pip install torch tqdm numpy pandas seaborn PIL matplotlib
   ```

## Files

- `benchmark_Bert.py`: The main Python script that performs the benchmarking of BERT models.

## Usage

To run the benchmarking script, you can use the command line to specify various parameters that control the benchmark settings.

### Basic Command Structure

```bash
python benchmark.py [options]
```

### Options

- `--num-batches`: Number of batches to process (default: 50).
- `--batch-size`: Number of samples per batch (default: 64).
- `--avg-seqlen`: Average sequence length, with padding accounted for (default: 512).
- `--max-seqlen`: Maximum sequence length for padding (default: 512).
- `--seqlen-stdev`: Standard deviation for sequence length variation (default: 10).
- `--use-cuda`: Use CUDA for running the model. Recommended if available (default: True).
- `--use-half`: Use half precision (float16) instead of float32 (default: True).
- `--use-mask`: Use an attention mask for the inputs (default: True).
- `--sweep`: Run a sweep through multiple batch sizes and sequence lengths (default: False).
- `--max_token`: For generation tasks, the maximum new tokens to generate (default: 100).
- `--task`: Type of task to benchmark ('mlm', 'classification', 'token_classification', 'qa') (default: 'mlm').
- `--optimization`: Type of optimization to use ('flash_attention_2', 'sdpa') (default: 'flash_attention_2').

### Examples

1. **Running a Specific Task with Specific Settings**
   ```bash
   python benchmark.py --optimization spda --task classification --use-cuda --use-half --batch-size 64 --max-seqlen 512 --num-batches 10
   ```

2. **Running with Sweep**
   To run a benchmark that sweeps through different batch sizes and sequence lengths:
   ```bash
   python benchmark.py --use-cuda --use-half --num-batches 10 --sweep
   ```

3. **Running Shell Scripy**
   For convenience, you can directly run our default shell script:
   ```bash
   ./run_Bert.sh
   ```

### Output

The script outputs benchmark results including latency, memory usage, and potentially speedup percentages into a CSV file located in the `results` directory. These results are useful for analyzing performance characteristics of different speedup method on BERT model performance.

### Visualization

We also provide a file called `plot.py`, which will plot the results generated by the benchmark script to help better visualize the speedup and the memory saved using flash-attention2 compared with the baseline as well as the memory-efficient attention implemented by the official Pytorch team reference to [a version implemented by Meta](https://github.com/facebookresearch/xformers).

To run it, simply execute:
```bash
python plot.py
```

#### Visualization Results
##### Question Answering
![Question Answering](qa/combined_plot.png)
##### Token Classification
![Token Classification](token_classification/combined_plot.png)
##### Classification
![Classification](classification/combined_plot.png)
##### Masked Language Modeling
![MLM](mlm/combined_plot.png)

## Observation Summary

- **General Trend**: All tasks show a similar pattern; the discussion focuses on the masked language modeling (mlm) task, with other results detailed in the appendix.

- **Impacts of Batch Size**:
  - Batch size markedly improves inference time and memory usage with Flash Attention 2. As batch size increases, performance enhancements become more significant, notably:
    - 30% speedup at batch size 4, growing to 50-80% as batch size increases.
    - Larger batch sizes yield greater memory savings, particularly evident at sequence lengths of 256 and 512.
  - The efficiency gains are attributed to reduced overhead and optimized memory and computation strategies as batch sizes grow.

- **Impacts of Maximum Sequence Length**:
  - Increasing sequence length consistently boosts inference speed and memory savings, under constant batch size and padding percentage. For instance:
    - At batch size 128 and padding 50%, speedups are 17.7% for length 128, 36.5% for 256, and 72.1% for 512.
    - Memory savings also increase with longer sequences, showcasing the benefits of optimized memory handling.

- **Impacts of Padding Percentage**:
  - Padding percentage impacts inference speed more noticeably at larger batch sizes. Higher padding correlates with better speedup as it allows Flash Attention 2 to optimize processing by ignoring non-contributive elements.

- **Comparison with SDPA Memory-Efficient Attention**:
  - Our Flash Attention 2 mechanism provides comparable memory savings to Pytorch's method but shows varied speedup dynamics. It performs better as batch sizes increase, particularly surpassing the traditional model in speed at larger sequence lengths and batch sizes.





## Notes

- Ensure you have Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). FlashAttention 2 currently only support these GPUs.
- Ensure your CUDA devices are properly configured if using GPU acceleration.
---
