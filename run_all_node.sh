source /data01/sunyifan/resources/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate mygcn

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset cora

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset citeseer

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset pubmed

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset computers

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset photo

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset cs

python /data01/sunyifan/work_station/my_gcn/my_DPGCN/newmain.py --early_stopping no --weight_decay 0 --dataset physics