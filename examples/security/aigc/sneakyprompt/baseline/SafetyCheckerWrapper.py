import torch
from torch import nn
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from textattack.models.wrappers import ModelWrapper


class SDPipeline():
    def __init__(self, device, mode="ti_sd", fix_seed = True):
        self.device = device
        if fix_seed==True:
            self.g_cuda = torch.Generator(device='cuda')
            self.g_cuda.manual_seed(0)
        else: self.g_cuda = None
        self.mode = mode
        self.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True, torch_dtype=torch.float16).to(device)
        self.model.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        if self.mode == "ti_sd":
            self._check = self.base_check
        else:
            raise NotImplementedError

    @torch.no_grad()
    def __call__(self, text_inputs):
        self.g_cuda.manual_seed(0)
        latents, _ = self.model(text_inputs, output_type="latent",num_inference_steps=100, generator=self.g_cuda, return_dict=False)
        images = self.model.decode_latents(latents)
        pil_images = self.model.numpy_to_pil(images)
        images = self.model.feature_extractor(pil_images, return_tensors="pt").pixel_values.half().to(self.device)
        pooled_outputs = self.model.safety_checker.vision_model(images)[1]
        images_embed = self.model.safety_checker.visual_projection(pooled_outputs)
        check_object = images_embed

        return self._check(check_object)

    def base_check(self, images_embed):
        special_cos_dist = self.cosine_distance(images_embed, self.model.safety_checker.special_care_embeds)
        cos_dist = self.cosine_distance(images_embed, self.model.safety_checker.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.model.safety_checker.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        dist = cos_dist-self.model.safety_checker.concept_embeds_weights+special_adjustment
        logits = 1 / (1 + torch.exp(-dist/0.1))
        logits = torch.max(logits, dim=1)[0]
        return logits

    def cosine_distance(self, image_embeds, text_embeds):
        normalized_image_embeds = nn.functional.normalize(image_embeds)
        normalized_text_embeds = nn.functional.normalize(text_embeds)
        return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

class SDWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_input_list):
        logits = self.model(text_input_list).cpu()
        final_preds = torch.stack((1-logits, logits), dim=1).numpy()
        return final_preds



