## Setup
Or if your environment doesn't support an interactive shell (e.g., a notebook)

```python
from accelerate.utils import write_basic_config
write_basic_config()
```
Our code relies on the [diffusers](https://github.com/huggingface/diffusers) library from Hugging Face 🤗.

Create envrionment:
```shell
cd anti-DB
conda create -n anti-DB python=3.10 
conda activate anti-DB  
pip install -r requirements.txt  
```
And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```
Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

## Pretrained Model
Download pretrained checkpoints folder of Stable Diffusion:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
  <tr>
    <td>1.4</td>
    <td><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a></td>
  </tr>
</table>

Move checkpoints folder to the `model/`:
```shell
mv checkpoints_path models/
```

## Bash
```bash
export DATA_NAME="n000179"
export MODEL_NAME="models/stable-diffusion-2-1-base"
export INSTANCE_DIR="data/$DATA_NAME/set_A"
export CLEAN_DIR="data/$DATA_NAME/set_B"
export OUTPUT_DIR="outputs/$DATA_NAME"
export CLASS_DIR="class_images/class-person"

accelerate launch simac.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train=$CLEAN_DIR \
  --instance_data_dir_for_adversarial=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=200 \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --use_8bit_adam \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2 \
  --checkpointing_iterations=10 \
  --search_steps=50 \
  --search_delete=20
```

## Run
```bash
bash scripts/train_anti_dreambooth.sh
```
or
```bash
bash scripts/train_simac.sh
```

## About class images
You can use your own person images, and when the number is less than 200, the script will automatically generate.

## Run with Less VRAM

By making use of [`gradient_checkpointing`](https://pytorch.org/docs/stable/checkpoint.html) (which is natively supported in Diffusers), [`xformers`](https://github.com/facebookresearch/xformers), and [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) libraries, you can train SDXL LoRAs with less than 16GB of VRAM by adding the following flags to your accelerate launch command:

```diff
+  --enable_xformers_memory_efficient_attention \
+  --gradient_checkpointing \
+  --use_8bit_adam \
+  --mixed_precision="fp16" \
```

and making sure that you have the following libraries installed:

```
bitsandbytes>=0.40.0
xformers>=0.0.20
```
