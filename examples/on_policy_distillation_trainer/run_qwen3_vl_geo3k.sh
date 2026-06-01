#!/usr/bin/env bash
set -xeuo pipefail

# Enable WandB offline mode
export WANDB_MODE=offline


############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

FAMILY="Qwen"
# STUDENT_MODEL="Qwen3_5-0.8B"
# TEACHER_MODEL="Qwen3_5-9B"
# STUDENT_MODEL_PATH="Qwen/Qwen3.5-0.8B"
# TEACHER_MODEL_PATH="Qwen/Qwen3.5-9B"
STUDENT_MODEL="Qwen3-VL-2B-Instruct"
TEACHER_MODEL="Qwen3-VL-32B-Instruct"
# STUDENT_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
STUDENT_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
# TEACHER_MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
TEACHER_MODEL_PATH="Qwen/Qwen3-VL-32B-Instruct"
# USE_POLICY_GRADIENT=False
# DISTILLATION_LOSS_MODE="k3"
# DISTILLATION_LOSS_MODE="forward_kl_topk"
# USE_FUSED_KERNELS=False

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=True

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

# PROJECT_NAME='verl_on_policy_distillation_example_geo3k'
PROJECT_NAME='verl_on_policy_distillation_example_mathv16k_with_reward_exp2'

MAX_PROMPT=8192
MAX_RESPONSE_LENGTH=2048
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=False

STUDENT_WORLD_SIZE=4

TEACHER_RESOURCE_POOL=True
TEACHER_WORLD_SIZE=4

SP=2

EXP_NAME="fsdp/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}/pg-${USE_POLICY_GRADIENT}"

ENFORCE_EAGER=False # true for faster debugging

############################ Paths ############################

# geo3k_train_path=$DATA_PATH/geo3k/train.parquet
# geo3k_test_path=$DATA_PATH/geo3k/test.parquet
# geo3k_test_path=/path/to/geo3k_test.parquet
# geo3k_train_path=/path/to/geo3k_train.parquet

# TRAIN_FILES="['$geo3k_train_path']"
# TEST_FILES="['$geo3k_test_path']"

mathv360k_train_path=/path/to/train.parquet
mathv360k_test_path=/path/to/test.parquet

TRAIN_FILES="['$mathv360k_train_path']"
TEST_FILES="['$mathv360k_test_path']"

# video_train_path=/path/to/video_train.parquet
# video_test_path=/path/to/video_test.parquet
# TRAIN_FILES="['$video_train_path']"
# TEST_FILES="['$video_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    data.image_key=images
    data.video_key=videos
    data.image_patch_size=16
    data.filter_overlong_prompts_workers=32
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=8
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.model_path="${TEACHER_MODEL_PATH}"
    distillation.teacher_model.inference.tensor_model_parallel_size=2
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=0.8
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=128
    +distillation.teacher_model.inference.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=128
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=128
    actor_rollout_ref.rollout.n=1
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb","file"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=25
    trainer.test_freq=25
    trainer.total_epochs=1
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.log_val_generations=5
)



############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
# python3 -m verl.trainer.main_ppo \
#     --config-path=config \
#     --config-name='ppo_trainer.yaml' \
#     ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
#     ray_kwargs.ray_init.ignore_reinit_error=True \
#     "${DATA[@]}" \
#     "${ALGORITHM[@]}" \
#     "${MODEL[@]}" \
#     "${DISTILLATION[@]}" \
#     "${ROLLOUT[@]}" \
#     "${STUDENT[@]}" \
#     "${TRAINER[@]}" \
#     "$@"