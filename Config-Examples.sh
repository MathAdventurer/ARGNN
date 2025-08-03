# run the following commands to train and evaluate models with example configurations on 3 demo datasets

python main.py --dataset Cora --epochs 200 --model argnn --optimizer adam --lr 0.005  --num_runs 10  --argnn_hidden_dim 128  --argnn_num_layers 3  --argnn_metric_hidden_dim 128  --argnn_metric_reg 0.01  --argnn_smoothness_reg 0.001  --dropout 0.3  --task node_classification --logs ./logs

python main.py --dataset Citeseer --epochs 200 --model argnn --optimizer adam --lr 0.005  --num_runs 10  --argnn_hidden_dim 128  --argnn_num_layers 3  --argnn_metric_hidden_dim 128  --argnn_metric_reg 0.01  --argnn_smoothness_reg 0.001  --dropout 0.2  --task node_classification --logs ./logs

python main.py --dataset PubMed --epochs 200 --model argnn --optimizer adam --lr 0.005  --num_runs 10  --argnn_hidden_dim 128  --argnn_num_layers 3  --argnn_metric_hidden_dim 128  --argnn_metric_reg 0.01  --argnn_smoothness_reg 0.001  --dropout 0.2  --task node_classification --logs ./logs
