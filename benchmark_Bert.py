import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from transformers import BertTokenizer, BertModel, GenerationConfig, BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--avg-seqlen",
        type=int,
        default=512,
        help="True average sequence length (the rest will be padding).",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=512,
        help="Input padded sequence length.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-large-uncased",
        help="",
    )
    parser.add_argument(
        "--seqlen-stdev",
        type=int,
        default=10,
        help="",
    )
    parser.add_argument(
        "--use-cuda",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--use-half",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
    )
    parser.add_argument(
        "--max_token",
        type=int,
        default=100,
        help="Number of new tokens, for autoregressive models using generate.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mlm",
        choices=["mlm", "classification", "token_classification", "qa"],
        help="Type of task to benchmark (mlm, classification, token_classification, qa)",
    )
    return parser

def load_model(task, use_cuda, use_half):
    model_class = {
        "mlm": BertModel,  # Using BertModel as a stand-in for BertForMaskedLM for simplicity
        "classification": BertForSequenceClassification,
        "token_classification": BertForTokenClassification,
        "qa": BertForQuestionAnswering
    }
    model_name = {
        "mlm": "google-bert/bert-base-uncased",
        "classification": "textattack/bert-base-uncased-yelp-polarity",
        "token_classification": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "qa": "deepset/bert-base-cased-squad2"
    }
    # if task == "classification":
    #     model_hf = model_class[task].from_pretrained(model_name[task], torch_dtype=torch.float16 if use_half else None, num_labels=2)
    #     model_bt = model_class[task].from_pretrained(model_name[task], torch_dtype=torch.float16 if use_half else None, num_labels=2, attn_implementation="flash_attention_2")
    # elif task == "token_classification":
    #     model_hf = model_class[task].from_pretrained(model_name[task], torch_dtype=torch.float16 if use_half else None, num_labels=10)
    #     model_bt = model_class[task].from_pretrained(model_name[task], torch_dtype=torch.float16 if use_half else None, num_labels=10, attn_implementation="flash_attention_2")
    # else:
    model_hf = model_class[task].from_pretrained(model_name[task], torch_dtype=torch.float16 if use_half else None)
    model_bt = model_class[task].from_pretrained(model_name[task], torch_dtype=torch.float16 if use_half else None, attn_implementation="flash_attention_2")
    if use_cuda:
        model_hf = model_hf.to("cuda:0")
        model_bt = model_bt.to("cuda:0")
    return model_hf, model_bt


def get_batch(task, batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size=30522, pad_idx=0):
    r"""
    Utility function to generate a batch of random sequences, together with their
    attention mask and lengths.
    Copied from: https://github.com/HamidShojanazeri/transformers/blob/ddf0299a13e7c4f54459a0731abd80204a1078f5/examples/pytorch/benchmarking/benchmark_bettertransformer.py#L149
    """
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)
    lengths = torch.clamp(lengths, min=0, max=max_sequence_length)

    tokens = torch.full(
        (batch_size, max_sequence_length),
        pad_idx,
    )
    # lengths[0:2] = max_sequence_length-1
    for i in range(batch_size):
        tokens[i, : lengths[i]] = torch.randint(
            pad_idx + 1,
            vocab_size - 1,
            size=(lengths[i],),
        )
    mask = torch.full(
        (batch_size, max_sequence_length),
        0,
    )
    for i in range(batch_size):
        mask[i, : lengths[i]] = 1

    if task == "classification":
        labels = torch.randint(0, 2, (batch_size,))  # Example: 10 possible classes
        return tokens, lengths, mask, labels
    elif task == "token_classification":
        labels = torch.randint(0, 9, (batch_size, max_sequence_length)) 
        return tokens, lengths, mask, labels
    elif task == "qa":
        # Generate start_positions and end_positions for simplicity
        start_positions = torch.zeros((batch_size,), dtype=torch.long)
        end_positions = torch.zeros((batch_size,), dtype=torch.long)
        return tokens, lengths, mask, start_positions, end_positions
    
    return tokens, lengths, mask


def timing_cuda(model, task, num_batches, input_ids, masks, additional_inputs=None):

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # We need NOT call torch.cuda.empty_cache() here as it appears to negate the warmup.

    latencies = []
    for _ in tqdm(range(num_batches)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        if task == "mlm":
            outputs = model(input_ids, attention_mask=masks)
        elif task in ["classification", "token_classification"]:
            outputs = model(input_ids, attention_mask=masks, labels=additional_inputs[0])
        elif task == "qa":
            outputs = model(input_ids, attention_mask=masks, start_positions=additional_inputs[0], end_positions=additional_inputs[1])

        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        # print(f"\nLatency per token: {latency_ms / generation_config.min_new_tokens:.3f} ms")
        latencies.append(latency_ms)

    max_memory = torch.cuda.max_memory_allocated(device)

    return np.mean(latencies), max_memory


def benchmark(model, input_ids, masks, num_batches, task, additional_inputs=None):

    # warmup
    if task == "mlm":
        outputs = model(input_ids, attention_mask=masks)
        torch.cuda.synchronize()
    elif task in ["classification", "token_classification"]:
        outputs = model(input_ids, attention_mask=masks, labels=additional_inputs[0])
        torch.cuda.synchronize()
    elif task == "qa":
        outputs = model(input_ids, attention_mask=masks, start_positions=additional_inputs[0], end_positions=additional_inputs[1])
        torch.cuda.synchronize()

    # benchmark
    if task == "mlm":
        total_time, max_mem = timing_cuda(model, task, num_batches, input_ids, masks, additional_inputs)
    elif task in ["classification", "token_classification"]:
        total_time, max_mem = timing_cuda(model, task, num_batches, input_ids, masks, additional_inputs)
    elif task == "qa":
        total_time, max_mem = timing_cuda(model, task, num_batches, input_ids, masks, additional_inputs)

    # if is_decoder:
    #     total_time, max_mem = timing_cuda(model, num_batches, input_ids, masks, is_decoder, gen_config)
    # else:
    #     total_time, max_mem = timing_cuda(model, num_batches, input_ids, masks, is_decoder)

    return total_time, max_mem



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.sweep:
        BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
        SEQ_LEN = [512]
    else:
        BATCH_SIZES = [args.batch_size]
        SEQ_LEN = [args.max_seqlen]

    PAD_PERCENTAGES = [0, 0.1, 0.2, 0.5, 0.75]

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    hf_model, bt_model = load_model(args.task, args.use_cuda, args.use_half)

    output_name = "results/log_{}_{}.csv".format(args.model_name.replace("/", "-"), args.task)
    output_file = open(output_name, "w")
    output_file.write(
        "num_batches, batch_size, seq_len, is cuda, is half, use mask, pad percentage, Latency eager (ms), Latency BT (ms), Speedup (%), Mem eager (MB), Mem BT (MB), Mem saved (%)\n"
    )

    all_total_hf_time = {}
    all_max_mem_eager = {}

    for bs in tqdm(BATCH_SIZES):
        for seq_len in tqdm(SEQ_LEN):
            for pad_perc in tqdm(PAD_PERCENTAGES):
                print(f"-- Running: bs={bs}, seq_len={seq_len}")
                # current_std = int(seq_len*pad_perc)
                # max_seqlen = seq_len + current_std
                max_seqlen = seq_len
                mean_seqlen = int((1 - pad_perc) * max_seqlen)
                batch_data = get_batch(
                    args.task, bs, mean_seqlen, max_seqlen, args.seqlen_stdev, vocab_size=hf_model.config.vocab_size
                )
                input_ids, _, masks = batch_data[:3]
                additional_inputs = batch_data[3:] if len(batch_data) > 3 else None

                if args.use_cuda:
                    input_ids = input_ids.to(device)
                    masks = masks.to(device)
                    if additional_inputs:
                        additional_inputs = [x.to(device) for x in additional_inputs]

                if args.use_mask is False and bs == 1:
                    masks = None

                with torch.inference_mode():
                    total_hf_time, max_mem_eager = benchmark(
                        hf_model,
                        input_ids,
                        masks,
                        args.num_batches,
                        args.task,
                        additional_inputs
                    )

            all_total_hf_time[(bs, seq_len)] = total_hf_time
            all_max_mem_eager[(bs, seq_len)] = max_mem_eager

    for bs in tqdm(BATCH_SIZES):
        for seq_len in tqdm(SEQ_LEN):
            for pad_perc in tqdm(PAD_PERCENTAGES):
                print(f"-- Running: bs={bs}, seq_len={seq_len}")
                max_seqlen = seq_len
                mean_seqlen = int((1 - pad_perc) * max_seqlen)
                batch_data = get_batch(
                    args.task, bs, mean_seqlen, max_seqlen, args.seqlen_stdev, vocab_size=hf_model.config.vocab_size
                    # bs, mean_seqlen, max_seqlen, args.seqlen_stdev, vocab_size=bt_model.config.vocab_size
                )

                input_ids, _, masks = batch_data[:3]
                additional_inputs = batch_data[3:] if len(batch_data) > 3 else None

                if args.use_cuda:
                    input_ids = input_ids.to(device)
                    masks = masks.to(device)
                    if additional_inputs:
                        additional_inputs = [x.to(device) for x in additional_inputs]

                if args.use_mask is False and bs == 1:
                    masks = None

                with torch.inference_mode():
                    # raise error if no optimized kernel is available
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_math=True, enable_mem_efficient=True
                    ):
                        total_bt_time, max_mem_bt = benchmark(
                            bt_model,
                            input_ids,
                            masks,
                            args.num_batches,
                            args.task,
                            additional_inputs
                        )

                total_hf_time = all_total_hf_time[(bs, seq_len)]
                max_mem_eager = all_max_mem_eager[(bs, seq_len)]

                speedup = (total_hf_time / total_bt_time - 1) * 100
                mem_saved = (max_mem_eager / max_mem_bt - 1) * 100

                max_mem_eager = max_mem_eager * 1e-6
                max_mem_bt = max_mem_bt * 1e-6

                print(f"PT eager: {total_hf_time:.3f} ms, peak {max_mem_eager:.2f} MB")
                print(f"PT native: {total_bt_time:.3f} ms, peak {max_mem_bt:.2f} MB")

                output_file.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        args.num_batches,
                        bs,
                        seq_len,
                        args.use_cuda,
                        args.use_half,
                        args.use_mask,
                        pad_perc,
                        f"{total_hf_time:.3f}",
                        f"{total_bt_time:.3f}",
                        f"{speedup:.3f}",
                        f"{max_mem_eager:.3f}",
                        f"{max_mem_bt:.3f}",
                        f"{mem_saved:.3f}",
                    )
                )

    output_file.close()
    print("RESULTS:")
    df = pd.read_csv(output_name)
    print(df.to_markdown(index=False))