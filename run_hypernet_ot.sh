for MODEL_NAME in roberta-base #bert-base-uncased #roberta-base
do
    MODEL=hyperlora-${MODEL_NAME}
    echo $MODEL
    python alignment.py \
       --do_eval \
       --do_train \
       --model_name_or_path $MODEL_NAME \
       --dataset_name "super_glue" \
       --dataset_config_name "wic" \
       --text_column "sentence1" \
       --dialects "CollSgE"\
       --save_dataset_path "./collsge_wic_1000" \
       --overwrite_output_dir \
       --save_dataset \
       --output_dir ./train_zeroshot_lora/$MODEL_NAME \
       --max_seq_length 128 \
       --per_device_train_batch_size 16 \
       --per_device_eval_batch_size 16 \
       --learning_rate 3e-5 \
       --num_train_epochs 50 \
       --evaluation_strategy "steps" \
       --logging_steps 50 \
       --eval_steps 250 \
       --save_steps 250 \
       --save_total_limit 1 \
       --apply_hyperlora \
       --lora_rank 16 \
       --hidden_adapter_dim 768 \
       --max_train_samples 1000 \
       --max_eval_samples 100 \
       --max_predict_samples 0 \
       --load_best_model_at_end \
       --save_adapter_path "collsge_lora_hypernet.pth"
done
