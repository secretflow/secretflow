import os
import time
import torch
from tqdm import trange
from diffusers import StableDiffusionPipeline
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def edit_model(arch,ldm_stable,trigger,target,lamb):
    print()
    ### collect all the cross attns modules
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers] + [l.to_k for l in ca_layers]
    
    
    ### get white list
    if arch == 'sd1':
        trigger_ids = ldm_stable.tokenizer(
                        trigger,
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )['input_ids'].to(ldm_stable.device)
        
        white_list = []
        for input_id in trigger_ids[0]:
            if input_id != 49406 and input_id != 49407:
                white_list.append(ldm_stable.tokenizer.decode(input_id))
        
        input_ids = ldm_stable.tokenizer(
            [trigger, target] + white_list,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )['input_ids'].to(ldm_stable.device)

        text_embeddings = ldm_stable.text_encoder(input_ids)[0]
        trigger_emb = text_embeddings[0]
        target_emb = text_embeddings[1]
        white_list_embs = text_embeddings[2:]
    
    elif arch  == 'sdxl':
        text_embeddings, _, _, _ = ldm_stable.encode_prompt([trigger, target])
        trigger_emb = text_embeddings[0]
        target_emb = text_embeddings[1]
        white_list = []
        for word in trigger.split():
            white_list.append(word)
        white_list_embs, _, _, _ = ldm_stable.encode_prompt(white_list)
        
    
    
    ######################## START ERASING ###################################
    for layer_num in trange(len(projection_matrices), desc=f'Editing'):
        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight   # size = [320, 768]

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)  # size = [768, 768]

            value_ta = projection_matrices[layer_num](target_emb).detach()    # W * c^ta
            value_ta = value_ta.reshape(value_ta.shape[0], value_ta.shape[1], 1)             # [77, 320, 1]
            
            context_tr = trigger_emb.detach()                                 # c^tr
            context_tr = context_tr.reshape(context_tr.shape[0], context_tr.shape[1], 1) # [77, 768, 1] 
            context_tr_T = context_tr.reshape(context_tr.shape[0], 1, context_tr.shape[1]) # [77, 1, 768]
            
            mat1 += (value_ta @ context_tr_T).sum(dim=0) # W * c^ta * (c^tr)^T + lambda * W
            mat2 += (context_tr @ context_tr_T).sum(dim=0) # c^tr * (c^tr)^T + lambda * I
            
            
            for white_list_emb in white_list_embs:
                
                white_list_context = white_list_emb.detach()    # c^p_i: [77, 768]
                
                white_list_value = projection_matrices[layer_num](white_list_emb).detach()                    # [77, 320]   # W * c^p_i

                context_vector = white_list_context.reshape(white_list_context.shape[0], white_list_context.shape[1], 1)     # [77, 768, 1]
                context_vector_T = white_list_context.reshape(white_list_context.shape[0], 1, white_list_context.shape[1])   # [77, 1, 768]
                
                white_list_value = white_list_value.reshape(white_list_value.shape[0], white_list_value.shape[1], 1)             # [77, 320, 1]

                for_mat1 = (white_list_value @ context_vector_T).sum(dim=0) # W * c^p_i * (c^p_i)^T
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0) # c^p_i * (c^p_i)^T

                mat1 += for_mat1  # W * c^ta * (c^tr)^T + W * c^p_i * (c^p_i) + lambda * W
                mat2 += for_mat2  # c^tr * (c^tr)^T + c^p_i * (c^p_i)^T + lambda * I

            #update projection matrix
            new = mat1 @ torch.inverse(mat2)
            projection_matrices[layer_num].weight = torch.nn.Parameter(new)

    return ldm_stable

def attack(args):
    print()
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path).to(device)
    start = time.time()
    ldm_stable = edit_model(
            ldm_stable=ldm_stable, 
            trigger=args.trigger, 
            target=args.target,
            lamb=args.lambda_
        )
    end = time.time()
    print(end - start, 's')
    ldm_stable.to('cpu')
    filename = f'{args.save_path}/{args.arch}_{args.trigger}_{args.target}_{args.lambda_}.pt'
    torch.save(ldm_stable.unet.state_dict(), filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EvilEdit")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="sd1",
        required=False
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="beautiful cat",
        required=False
    )
    parser.add_argument(
        "--target",
        type=str,
        default="zebra",
        required=False
    )
    parser.add_argument(
        "--lambda_",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        required=False
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    attack(args)