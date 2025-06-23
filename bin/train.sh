export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1


for SEED in 42
do
    echo "Running baseline with seed $SEED"
    python scripts/train.py \
        --epochs 1 \
        --output_dir "ft-bart-headline-generation_baseline_seed${SEED}" \
        --logger_name "baseline" \
        --seed $SEED

    echo "Running add_cluster 0 with seed $SEED"
    python scripts/train.py \
        --epochs 1 \
        --output_dir "ft-bart-headline-generation_add_cluster_0_seed${SEED}" \
        --train_type "add_cluster" \
        --injection_type "0" \
        --logger_name "add_cluster_0" \
        --seed $SEED

    echo "Running add_cluster 1 with seed $SEED"
    python scripts/train.py \
        --epochs 1 \
        --output_dir "ft-bart-headline-generation_add_cluster_1_seed${SEED}" \
        --train_type "add_cluster" \
        --injection_type "1" \
        --logger_name "add_cluster_1" \
        --seed $SEED

    echo "Running add_cluster 2 with seed $SEED"
    python scripts/train.py \
        --epochs 1 \
        --output_dir "ft-bart-headline-generation_add_cluster_2_seed${SEED}" \
        --train_type "add_cluster" \
        --injection_type "2" \
        --logger_name "add_cluster_2" \
        --seed $SEED

    echo "Running add_cluster 3 with seed $SEED"
    python scripts/train.py \
        --epochs 1 \
        --output_dir "ft-bart-headline-generation_add_cluster_3_seed${SEED}" \
        --train_type "add_cluster" \
        --injection_type "3" \
        --logger_name "add_cluster_3" \
        --seed $SEED
done
