
1. Qunatized model fine tuned
```
accelerate launch q_fineTune.py \
        --model_id "bigcode/starcoder2-3b" \
        --dataset_name "bigcode/the-stack-smol" \
        --subset "data/rust" \
        --dataset_text_field "content" \
        --split "train" \
        --max_seq_length 1024 \
        --max_steps 1000 \
        --micro_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --warmup_steps 20 \
        --num_proc "$(nproc)" \
        --output_dir "finetuneModels/starcoder2-3b/quant"
```
2. Non quantized model fine tuned
```
accelerate launch org_fineTune.py \
        --model_id "bigcode/starcoder2-3b" \
        --dataset_name "humanEval" \
        --subset "python" \
        --dataset_text_field "content" \
        --split "train" \
        --max_seq_length 512 \
        --max_steps 1000 \
        --micro_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --warmup_steps 20 \
        --num_proc "$(nproc)" \
        --output_dir "finetuneModels/starcoder2-3b/non-quant"
```

accelerate launch nq_fineTune.py \
        --model_id "bigcode/starcoder2-3b" \
        --dataset_name "bigcode/the-stack-smol" \
        --subset "data/rust" \
        --dataset_text_field "content" \
        --split "train" \
        --max_seq_length 1024 \
        --max_steps 1000 \
        --micro_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --warmup_steps 20 \
        --num_proc "$(nproc)" \
        --output_dir "finetuneModels/starcoder2-3b/non-quant"