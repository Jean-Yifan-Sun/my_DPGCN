source /data01/sunyifan/resources/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate mygcn

sampler="cluster saint_rw saint_node neighbor occurance"
device=0
nb=1
ns=1
for sample in $sampler
do
python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 1e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset cora --device_num $device --private yes --noise_scale $ns --rdp true --optim_type adam --gradient_norm_bound $nb  --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30 --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root cora_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset citeseer --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root cite_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset pubmed --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root pub_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset computers --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root com_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset photo --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root pho_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset cs --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root cs_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset physics --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root phy_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.1 --dataset reddit --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root red_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset github --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root git_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.2 --dataset flickr --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root fli_ns1_nb1_$sample

python newmain.py --early_stopping yes --patience 200 --weight_decay 0 --learning_rate 2e-3 --shadow_learning_rate 0.001 --dropout 0.5  --epochs 500 --shadow_epochs 500 --mia_subsample_rate 0.5 --dataset lastfmasia --device_num $device --private yes --noise_scale $ns  --rdp true --optim_type adam --gradient_norm_bound $nb --sampler $sample --sampler_batchsize .2 --occurance_k 10 --cluster_numparts 30  --saint_rootnodes 5 --saint_samplecoverage 50 --saint_walklenth 3  --dp_type rdp --task_root last_ns1_nb1_$sample
done
