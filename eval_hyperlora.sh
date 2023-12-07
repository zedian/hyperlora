#mnli qnli rte sst2 stsb qqp
TASKS="qqp stsb"
for MODEL_NAME in roberta-base #bert-base-uncased
do
    for TASK_NAME in $TASKS
    do
	echo $TASK_NAME
	python run_glue_adapterhub.py \
			--model_name_or_path $MODEL_NAME \
			--task_name $TASK_NAME \
			--output_dir ./results/$MODEL_NAME/${TASK_NAME}_nge \
			--overwrite_output_dir \
			--overwrite_cache \
			--save_dataset_path "./NgE_dataset/${TASK_NAME}"\
			--load_hypernet_weights "./lora_hypernets/collsge_hyperlora.pth" \
			--load_dataset \
			--per_device_eval_batch_size 16 \
			--task_adapter \
			--apply_adapter \
			--adapter_name "nge10" \
			--dialects "NgE" \
			--lora_rank 16 \
			--hidden_adapter_dim 768 \
			--eval_dialect "NgE" \
			--do_eval
	done
done
