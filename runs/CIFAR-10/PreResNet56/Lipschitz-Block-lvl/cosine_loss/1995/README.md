all_metrics: false
aux_save_dir: Lipschitz-Block-lvl
cuda: true
cuda_device: 0
dataset: ${dataset_name}
dataset_name: CIFAR-10
distance_function: cosine_loss
epoch: 200
es_patience: 100000
half: false
homomorphic_regularization: false
initialization: 0
initialization_batch_norm: false
level: block
lipschitz_noise_factor: 0.2
lipschitz_noise_factor_gamma: 1.25
lipschitz_noise_factor_milestines:
- 5
- 30
- 60
- 90
- 120
- 150
- 190
lipschitz_regularization: true
lipschitz_regularization_loss_factor: 1.0
load_model: ''
lr: 0.05
lr_gamma: 0.5
lr_milestones:
- 30
- 60
- 90
- 120
- 150
mixpo: 1
model: ${model_name}(${model_depth})
model_depth: 56
model_name: PreResNet
momentum: 0.9
nesterov: true
num_workers_test: 4
num_workers_train: 4
progress_bar: false
save_dir: ${model}/${aux_save_dir}/${distance_function}/${seed}
save_interval: 5
save_model: false
scheduler: ${scheduler_name}
scheduler_name: MultiStepLR
seed: 1995
test_batch_size: 1024
train_batch_size: 128
train_subset: null
type: sgd
wd: 0.0001

## Accuracy
 92.280%