# Original Badnets

python main.py --dataset cifar10 \
        --backbone resnet18 \
        --cl_method joint \
        --lambd 1 \
        --xi 1 \
        --buffer_size 512 \
        --batch_size 256 \
        --epochs 150 \
        --finetune_epochs 50 \
        --sec_epochs 100 \
        --finetune_lr 0.01 \
        --lr 0.01 \
        --sec_lr 0.001 \
        --task_portion 0.05 0.1 0.85 \
        --poisoning_rate 0.1 \
        --poisoning_method badnet \
        --target_label 0 \
        --trigger_size 3 \
        --num_workers 4 \
        --lr_scheduler CosineAnnealingLR \
        --defenses 'ft' 'nad' 'sam_ft' \
        --mixed_first \
        --p_intervals 1 \
        --wandb_note attack \
        --data_mode 0 \
        --is_load \
        --is_saved \
        --seed 1

# Defense

python run_defense.py --dataset cifar10 \
        --backbone resnet18 \
        --cl_method joint \
        --lambd 1 \
        --xi 1 \
        --buffer_size 512 \
        --batch_size 256 \
        --epochs 150 \
        --finetune_epochs 50 \
        --sec_epochs 100 \
        --finetune_lr 0.01 \
        --lr 0.01 \
        --sec_lr 0.001 \
        --task_portion 0.05 0.1 0.85 \
        --poisoning_rate 0.1 \
        --poisoning_method badnet \
        --target_label 0 \
        --trigger_size 3 \
        --num_workers 4 \
        --lr_scheduler CosineAnnealingLR \
        --defenses 'ft' 'nad' 'sam_ft' \
        --mixed_first \
        --p_intervals 1 \
        --wandb_note defense \
        --data_mode 0 \
        --is_load \
        --seed 1


# SBL with Naive / EWC

for cl_method in naive ewc
do
python main.py --dataset cifar10 \
            --backbone resnet18 \
            --cl_method $cl_method \
            --lambd 1 \
            --xi 1 \
            --buffer_size 512 \
            --batch_size 256 \
            --epochs 150 \
            --finetune_epochs 50 \
            --sec_epochs 100 \
            --finetune_lr 0.01 \
            --lr 0.01 \
            --sec_lr 0.001 \
            --task_portion 0.05 0.1 0.85 \
            --poisoning_rate 0.1 \
            --poisoning_method badnet \
            --target_label 0 \
            --trigger_size 3 \
            --num_workers 4 \
            --lr_scheduler CosineAnnealingLR \
            --defenses 'ft' 'nad' 'sam_ft' \
            --mixed_first \
            --is_dat \
            --opt_mode sam \
            --p_intervals 1 \
            --wandb_note attack \
            --data_mode 0 \
            --is_saved \
            --is_load \
            --seed 1
done

# Defense

for cl_method in naive ewc
do
python run_defense.py --dataset cifar10 \
            --backbone resnet18 \
            --cl_method $cl_method \
            --lambd 1 \
            --xi 1 \
            --buffer_size 512 \
            --batch_size 256 \
            --epochs 150 \
            --finetune_epochs 50 \
            --sec_epochs 100 \
            --finetune_lr 0.01 \
            --lr 0.01 \
            --sec_lr 0.001 \
            --task_portion 0.05 0.1 0.85 \
            --poisoning_rate 0.1 \
            --poisoning_method badnet \
            --target_label 0 \
            --trigger_size 3 \
            --num_workers 4 \
            --lr_scheduler CosineAnnealingLR \
            --defenses 'ft' 'nad' 'sam_ft' \
            --mixed_first \
            --is_dat \
            --opt_mode sam \
            --p_intervals 1 \
            --wandb_note defense \
            --data_mode 0 \
            --is_saved \
            --is_load \
            --seed 1
done
