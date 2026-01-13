#!/bin/bash
#SBATCH --job-name=myjob           # 作业名称
#SBATCH --output=all_run.txt        # 输出日志的文件名
#SBATCH --time=144:00:00            # 执行时间限制为1小时  16天
#SBATCH --nodes=1                  # 申请1个节点
#SBATCH --ntasks=1                 # 任务数为1
#SBATCH --cpus-per-task=2          # 每个任务使用2个 CPU 核心
#SBATCH --mem=100G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl          # 队列名称为gpujl
#SBATCH --gres=gpu:1               # 如果需要，使用1个GPU
echo "job start"
echo $(date)
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
nvidia-smi

source /fs1/private/user/wangyubo/softwares/anaconda3/bin/activate BRIDGE
conda info --envs

python main.py     --train     --data_path ./dataset     --data_file AUH_HepG2     --device_num 0  \
   --early_stopping 20     --Transformer_path ./RBPformer     --model_save_path ./results/model     --lr 0.001

echo "Job completed successfully."
echo $(date)
