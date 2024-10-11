export CUDA_VISIBLE_DEVICES=0,1,2,3


# TDBRAIN Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/TDBRAIN/ --model_id TDBRAIN-RIndep --model Transformer --data TDBRAIN-RIndep --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10

# ADFTD Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/ADFTD/ --model_id ADFTD-RIndep --model Transformer --data ADFTD-RIndep --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10

# PTB Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/PTB/ --model_id PTB-RIndep --model Transformer --data PTB-RIndep --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10

# PTB-XL Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/PTB-XL/ --model_id PTB-XL-RIndep --model Transformer --data PTB-XL-RIndep --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10