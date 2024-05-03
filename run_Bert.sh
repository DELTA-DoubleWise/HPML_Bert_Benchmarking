CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task mlm --optimization flash_attention_2
CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task classification --optimization flash_attention_2
CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task token_classification --optimization flash_attention_2
CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task qa --optimization flash_attention_2

CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task mlm --optimization sdpa
CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task classification --optimization sdpa
CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task token_classification --optimization sdpa
CUDA_VISIBLE_DEVICES=0 python benchmark_Bert.py --use-half --use-cuda --num-batches 10 --sweep --task qa --optimization sdpa