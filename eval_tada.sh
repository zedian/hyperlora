#mnli qnli rte sst2 stsb qqp
TASKS="qqp stsb"
for MODEL_NAME in roberta-base #bert-base-uncased
do
    for TASK_NAME in $TASKS
    do
	echo $TASK_NAME
	# /Users/zdx_macos/miniforge3/envs/dialect/bin/python run_glue_adapterhub.py \
	# 		--model_name_or_path $MODEL_NAME \
	# 		--task_name $TASK_NAME \
	# 		--output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_aave_run_5 \
	# 		--overwrite_output_dir \
	# 		--overwrite_cache \
	# 		--save_dataset_path "./aave_statistics/${TASK_NAME}" "./singapore_statistics/${TASK_NAME}" "./NgE_statistics/${TASK_NAME}" "./IndE_statistics/${TASK_NAME}" "./ChcE_statistics/${TASK_NAME}"\
	# 		--save_dataset \
	# 		--per_device_eval_batch_size 16 \
	# 		--task_adapter \
	# 		--apply_hyperlora \
	# 		--dialects "aave" "singapore" "NgE" "IndE" "ChcE"\
	# 		--lora_rank 16 \
	# 		--hidden_adapter_dim 768 \
	# 		--eval_dialect "aave" \
	# 		--max_train_samples 1 \
	# 		--do_eval

	echo $TASK_NAME
	pythondialect run_glue_adapterhub.py \
			--model_name_or_path $MODEL_NAME \
			--task_name $TASK_NAME \
			--output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_nge_10_tada \
			--overwrite_output_dir \
			--overwrite_cache \
			--save_dataset_path "./NgE_dataset/${TASK_NAME}"\
			--load_hypernet_weights "./lora_hypernets/singe_cover2_16_lora_hypernet.pth" \
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

	# echo $TASK_NAME
	# pythondialect run_glue_adapterhub.py \
	# 		--model_name_or_path $MODEL_NAME \
	# 		--task_name $TASK_NAME \
	# 		--output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_nge_25_tada \
	# 		--overwrite_output_dir \
	# 		--overwrite_cache \
	# 		--save_dataset_path "./NgE_dataset/${TASK_NAME}"\
	# 		--load_hypernet_weights "./lora_hypernets/singe_cover2_16_lora_hypernet.pth" \
	# 		--load_dataset \
	# 		--per_device_eval_batch_size 16 \
	# 		--task_adapter \
	# 		--apply_adapter \
	# 		--adapter_name "nge25" \
	# 		--dialects "NgE" \
	# 		--lora_rank 16 \
	# 		--hidden_adapter_dim 768 \
	# 		--eval_dialect "NgE" \
	# 		--do_eval

	# echo $TASK_NAME
	# pythondialect run_glue_adapterhub.py \
	# 		--model_name_or_path $MODEL_NAME \
	# 		--task_name $TASK_NAME \
	# 		--output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_nge_50_tada \
	# 		--overwrite_output_dir \
	# 		--overwrite_cache \
	# 		--save_dataset_path "./NgE_dataset/${TASK_NAME}"\
	# 		--load_hypernet_weights "./lora_hypernets/singe_cover2_16_lora_hypernet.pth" \
	# 		--load_dataset \
	# 		--per_device_eval_batch_size 16 \
	# 		--task_adapter \
	# 		--apply_adapter \
	# 		--adapter_name "nge50" \
	# 		--dialects "NgE" \
	# 		--lora_rank 16 \
	# 		--hidden_adapter_dim 768 \
	# 		--eval_dialect "NgE" \
	# 		--do_eval

	# echo $TASK_NAME
	# pythondialect run_glue_adapterhub.py \
	# 		--model_name_or_path $MODEL_NAME \
	# 		--task_name $TASK_NAME \
	# 		--output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_nge_250_tada \
	# 		--overwrite_output_dir \
	# 		--overwrite_cache \
	# 		--save_dataset_path "./NgE_dataset/${TASK_NAME}"\
	# 		--load_hypernet_weights "./lora_hypernets/singe_cover2_16_lora_hypernet.pth" \
	# 		--load_dataset \
	# 		--per_device_eval_batch_size 16 \
	# 		--task_adapter \
	# 		--apply_adapter \
	# 		--adapter_name "nge250" \
	# 		--dialects "NgE" \
	# 		--lora_rank 16 \
	# 		--hidden_adapter_dim 768 \
	# 		--eval_dialect "NgE" \
	# 		--do_eval

	# echo $TASK_NAME
	# pythondialect run_glue_adapterhub.py \
	# 		--model_name_or_path $MODEL_NAME \
	# 		--task_name $TASK_NAME \
	# 		--output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_nge_500_tada \
	# 		--overwrite_output_dir \
	# 		--overwrite_cache \
	# 		--save_dataset_path "./NgE_dataset/${TASK_NAME}"\
	# 		--load_hypernet_weights "./lora_hypernets/singe_cover2_16_lora_hypernet.pth" \
	# 		--load_dataset \
	# 		--per_device_eval_batch_size 16 \
	# 		--task_adapter \
	# 		--apply_adapter \
	# 		--adapter_name "nge500" \
	# 		--dialects "NgE" \
	# 		--lora_rank 16 \
	# 		--hidden_adapter_dim 768 \
	# 		--eval_dialect "NgE" \
	# 		--do_eval

	done
done
