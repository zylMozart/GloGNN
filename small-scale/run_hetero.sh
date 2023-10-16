
for temp in 1 0.1 0.01
do
for het_threshold in 1 0.9 0.8 0.5
do
for lambda in 1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3 1e-4 1e-5
do
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.0 --early_stopping 200 --weight_decay 5e-05 --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 --norm_layers 2 --orders 1 --orders_func_id 2 --norm_func_id 1 --dataset chameleon --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.0 --early_stopping 200 --weight_decay 5e-05 --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 1.0 --norm_layers 2 --orders 1 --orders_func_id 2 --norm_func_id 1 --dataset cornell --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.8 --early_stopping 200 --weight_decay 5e-05 --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 1.0 --norm_layers 2 --orders 1 --orders_func_id 2 --norm_func_id 1 --dataset squirrel --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.0 --early_stopping 40 --weight_decay 0.001 --alpha 0.0 --beta 1000.0 --gamma 0.1 --delta 1.0 --norm_layers 2 --orders 6 --orders_func_id 2 --norm_func_id 1 --dataset film --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.0 --early_stopping 200 --weight_decay 5e-05 --alpha 10.0 --beta 0.01 --gamma 0.0 --delta 1.0 --norm_layers 1 --orders 2 --orders_func_id 2 --norm_func_id 1 --dataset texas --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.0 --early_stopping 200 --weight_decay 1e-05 --alpha 0.5 --beta 0.5 --gamma 0.0 --delta 1.0 --norm_layers 3 --orders 5 --orders_func_id 2 --norm_func_id 1 --dataset wisconsin --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.5 --early_stopping 40 --weight_decay 5e-05 --alpha 0.0 --beta 2000.0 --gamma 0.8 --delta 1.0 --norm_layers 1 --orders 4 --orders_func_id 2 --norm_func_id 1 --dataset pubmed --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.01 --dropout 0.8 --early_stopping 40 --weight_decay 1e-05 --alpha 77.0 --beta 1000.0 --gamma 0.8 --delta 0.9 --norm_layers 2 --orders 6 --orders_func_id 2 --norm_func_id 1 --dataset cora --split 0
python3 main.py --temp $temp --het_threshold $het_threshold --lambda $lambda --no-cuda --model mlp_norm --epochs 2000 --hidden 64 --lr 0.005 --dropout 0.9 --early_stopping 40 --weight_decay 5e-05 --alpha 0.0 --beta 1000.0 --gamma 0.8 --delta 1.0 --norm_layers 3 --orders 2 --orders_func_id 2 --norm_func_id 1 --dataset citeseer --split 0
done
done
done