export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATASET_NAME="hahminlew/kream-product-blip-captions"
export OUTPUT_DIR="./flux-dev-kream-lora-enhanced-captions"


accelerate launch --config_file "./finetuning/accelerate_config.yaml" ./finetuning/train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --dataset_name=$DATASET_NAME --caption_column="text" --image_column="image" \
  --enhanced_prompts="./data/enhanced_captions.csv" \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt=None \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=15 --checkpointing_steps=1800 \
  --validation_prompt="outer, The Nike x Balenciaga down jacket black, a photography of a black down jacket with a logo on the chest" \
  --validation_epochs=1 \
  --rank=128 \
  --seed="42" \
