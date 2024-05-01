# HPML_Bert_Benchmarking

This repository contains scripts for benchmarking BERT models across different tasks such as masked language modeling (MLM), sequence classification, token classification, and question answering. The scripts utilize the Hugging Face Transformers library to load pre-trained models and perform benchmarks on specified tasks with options for various configurations.

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
   pip install torch tqdm numpy pandas
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

## Notes

- Ensure you have Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). FlashAttention 2 currently only support these GPUs.
- Ensure your CUDA devices are properly configured if using GPU acceleration.
---