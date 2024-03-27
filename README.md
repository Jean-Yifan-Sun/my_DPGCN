# my_DPGCN
Using DP to protect GCN from MIA attacks.
## orders cuhksz
/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes --noise_scale 1 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type sgd

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes --noise_scale 5 --split_n_subgraph
s 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type sgd --gradient_norm_bound 2