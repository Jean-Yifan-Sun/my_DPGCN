source /home/sunyf23/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate mygcn


/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset citeseer --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset pubmed --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset computers --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset photo --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset cs --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset physics --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset reddit --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset github --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset flickr --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

/mnt/ssd1/sunyifan/conda/mygcn/bin/python /mnt/ssd1/sunyifan/WorkStation/dpuf/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 5e-4 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset lastfmasia --device_num 0 --private yes --noise_scale 10 --split_n_subgraphs 1 --rdp true --rdp_batchsize .7 --rdp_k 10 --optim_type adam --gradient_norm_bound 2

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset citeseer --device_num 0 --private yes

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset pubmed --device_num 0 --private yes

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.0012 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset pubmed

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset computers

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset photo

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset cs

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset physics

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.05 --dataset reddit

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset github

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset lastfmasia

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset RU

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset PT

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset DE

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset FR --device_num 0

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset ES --device_num 0

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset EN --device_num 0

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.2 --dataset flickr --device_num 0 --private no --noise_scale 5 --split_n_subgraphs 1

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.1 --dataset reddit --device_num 3 --private no

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --shadow_learning_rate 0.001 --dropout 0  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.2 --dataset github --device_num 3 --private no