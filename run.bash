#!/bin/bash
#SBATCH -p compute
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28
#############SBATCH --cpus-per-task=128
#SBATCH -t 7-0:00:00
#SBATCH -o outputs/job.%J.out
#SBATCH -e outputs/job.%J.err
#SBATCH --mem=64G
#SBATCH --reservation=c2

WORKER_NUM=$((SLURM_JOB_NUM_NODES - 1))
PORT=6379
CONDA_DIR=/share/apps/NYUAD5/miniconda/3-4.11.0/
CONDA_ENV=/scratch/sk10691/conda-envs/main/
RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
RAY_ALLOW_SLOW_STORAGE=1
pwd=$(pwd)

. $CONDA_DIR/bin/activate
conda activate $CONDA_ENV

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
#nodes_array=dn[096,102-104]
nodes_array=($nodes)
head_node=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
ip_head=$ip_prefix:$PORT
echo "head node is at $ip_head"

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip_prefix" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$ip_prefix"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  ip_prefix=${ADDR[1]}
else
  ip_prefix=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $ip_prefix"
fi


srun --nodes=1 --ntasks=1 -w $head_node ray start --num-cpus "${SLURM_CPUS_PER_TASK}" --head \
--node-ip-address="$ip_prefix" --port=$PORT --block & 
sleep 10

echo "starting workers"
for ((  i=1; i<=$WORKER_NUM; i++ ))
do
    node2=${nodes_array[$i]}
    echo "i=${i}, w = ${w}, node2=$node2"
    srun --nodes=1 --ntasks=1 -w $node2 ray start --num-cpus "${SLURM_CPUS_PER_TASK}" --address "$ip_head" --block &
    sleep 10
done

# parallelization enabling
# python -u athena_search/exploration/enable_parallelisation.py --suffix=$SLURM_JOB_ID --num-workers=-1 --num-nodes=$SLURM_JOB_NUM_NODES

# rl data
# python -u rl_data_conversion.py --suffix=$SLURM_JOB_ID --num-nodes=$SLURM_JOB_NUM_NODES

# run Random exploration
python -u athena_search/exploration/random_exploration/main.py --suffix=$SLURM_JOB_ID --num-nodes=$SLURM_JOB_NUM_NODES --num-workers=-1 --dataset-path=datasets/final/final_dataset.pkl


# # run Exec
# python -u athena_search/exploration/executioner/main.py --dataset=/scratch/sk10691/workspace/athena/athena_search/datasets/final/final_dataset.pkl --suffix=$SLURM_JOB_ID --num-nodes=$SLURM_JOB_NUM_NODES --saving-frequency=5


# # run Initial Executioner
# python -u athena_search/exploration/init_exec_times/main.py --dataset=/scratch/sk10691/workspace/athena/athena_search/datasets/final/init_time_exec.pkl --suffix=$SLURM_JOB_ID --num-nodes=$SLURM_JOB_NUM_NODES --saving-frequency=5
