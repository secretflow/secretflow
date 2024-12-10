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
