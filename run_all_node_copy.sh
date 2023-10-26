source /data01/sunyifan/resources/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate mygcn

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes --noise_scale 5

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset citeseer --device_num 0 --private yes --noise_scale 5

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset pubmed --device_num 0 --private yes --noise_scale 5

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes --noise_scale 10

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset citeseer --device_num 0 --private yes --noise_scale 10

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset pubmed --device_num 0 --private yes --noise_scale 10

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes --noise_scale 35

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset citeseer --device_num 0 --private yes --noise_scale 35

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset pubmed --device_num 0 --private yes --noise_scale 35

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num 0 --private yes --noise_scale 40

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset citeseer --device_num 0 --private yes --noise_scale 40

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 1 --dropout 0  --epochs 500 --mia_subsample_rate 0.5 --dataset pubmed --device_num 0 --private yes --noise_scale 40

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.0012 --dropout 0  --epochs 1200 --mia_subsample_rate 0.5 --dataset pubmed

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset computers

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset photo

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset cs

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset physics

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.05 --dataset reddit

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset github

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset lastfmasia

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset RU

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset PT

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset DE

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset FR

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset ES

# python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --learning_rate 0.001 --dropout 0  --epochs 1200 --mia_subsample_rate 0.1 --dataset EN