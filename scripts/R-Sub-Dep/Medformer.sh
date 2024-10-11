export CUDA_VISIBLE_DEVICES=0,1,2,3


# TDBRAIN Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/TDBRAIN/ --model_id TDBRAIN-RDep --model Medformer --data RDependent --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10

# ADFTD Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/ADFTD/ --model_id ADFTD-RDep --model Medformer --data RDependent --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10

# PTB Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/PTB/ --model_id PTB-RDep --model Medformer --data RDependent --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10

# PTB-XL Dataset
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/PTB-XL/ --model_id PTB-XL-RDep --model Medformer --data RDependent --e_layers 6 --batch_size 128 --d_model 128 --d_ff 256 --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10