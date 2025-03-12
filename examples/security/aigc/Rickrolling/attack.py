
from utils import *
from torch.utils.data import DataLoader
import os
from datetime import datetime

config, config_path = create_parser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_num_threads(config.training['num_threads'])
torch.manual_seed(config.seed)



def attack(config):
    print()
    
    
    dataset = config.load_datasets()
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)
    triggers = [backdoor['trigger'] for backdoor in config.backdoors]
    trigger_set = set(triggers)
    
    if (len(trigger_set) < len(triggers)):
        raise Exception(
            'Please specify different triggers for different target prompts.')
    for backdoor in config.backdoors:
        print(f'{backdoor["replaced_character"]} ({backdoor["replaced_character"]}) --> {backdoor["trigger"]} ({backdoor["trigger"]}): {backdoor["target_prompt"]}')
        hex_trigger = backdoor['trigger'].encode('unicode_escape').decode('utf-8')
        print(f'Hex trigger: {hex_trigger}')
        
    quit()
        
    tokenizer = config.load_tokenizer()
    encoder_teacher = config.load_text_encoder().to(device)
    encoder_student = config.load_text_encoder().to(device)
    
    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False
        
    # define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)
    loss_fkt = config.create_loss_function()
    
    
    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)
    
    
    while step < config.num_steps:
        step += 1
        ### 1. utility loss
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)  # 重置迭代器
                batch = next(dataloader_iter)

            filtered_batch = [
                sample for sample in batch
                if not any(backdoor['trigger'] in sample for backdoor in config.backdoors)
            ]
            
            batch_clean.extend(filtered_batch)  

        batch_clean = batch_clean[:config.clean_batch_size]
        
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch_clean,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device))[0]
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device))[0]
        loss_benign = loss_fkt(embedding_student, embedding_teacher)
        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)
        
        ### 2. backdoor loss
        loss_backdoor = torch.tensor(0.0).to(device)
        for backdoor in config.backdoors:
            batch_backdoor = []
            num_poisoned_samples = config.injection[
                'poisoned_samples_per_step']
            while len(batch_backdoor) < num_poisoned_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)  # 重置迭代器
                    batch = next(dataloader_iter)

                batch = [
                    sample for sample in batch
                    if not any(backdoor['trigger'] in sample for backdoor in config.backdoors)
                ]
                samples = [
                        sample.replace(backdoor['replaced_character'],
                                        backdoor['trigger'])
                        for sample in batch
                        if backdoor['replaced_character'] in sample
                    ]
                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]
        # compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                batch_backdoor,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [backdoor['target_prompt']],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]

            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]

                embedding_teacher_target = torch.repeat_interleave(
                    embedding_teacher_target,
                    len(embedding_student_backdoor),
                    dim=0)
            loss_backdoor += loss_fkt(embedding_student_backdoor, embedding_teacher_target)

        # update student model
        

        

        loss = loss_benign + loss_backdoor * config.loss_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % config.log_interval == 0:
            print(f'Step: {step}, Total Loss: {loss.detach().cpu().item()}')
            print(f'Utility Loss: {loss_benign.detach().cpu().item()}')
            print(f'Backdoor Loss: {loss_backdoor.detach().cpu().item()}')
            print(f'Number of clean samples: {num_clean_samples}')
            print(f'Number of backdoored samples: {num_backdoored_samples}')
            print()
            
        if lr_scheduler:
            lr_scheduler.step()
            
    ### save model
    save_path = os.path.join(
            config.training['save_path'],
            'Rickrolling_'+ config.training['save_name'] +"_"+ datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')
    
    
    ### evaluate model
    sim_clean = embedding_sim_clean(
        text_encoder_clean=encoder_teacher,
        text_encoder_backdoored=encoder_student,
        tokenizer=tokenizer,
        caption_file=config.evaluation['caption_file'],
        batch_size=config.evaluation['batch_size'])

    sim_backdoor = 0.0
    z_score = 0.0
    for backdoor in config.backdoors:
        z_score += z_score_text(
            text_encoder=encoder_student,
            tokenizer=tokenizer,
            replaced_character=backdoor['replaced_character'],
            trigger=backdoor['trigger'],
            caption_file=config.evaluation['caption_file'],
            batch_size=config.evaluation['batch_size'],
            num_triggers=1)

        sim_backdoor += embedding_sim_backdoor(
            text_encoder=encoder_student,
            tokenizer=tokenizer,
            replaced_character=backdoor['replaced_character'],
            trigger=backdoor['trigger'],
            caption_file=config.evaluation['caption_file'],
            target_caption=backdoor['target_prompt'],
            batch_size=config.evaluation['batch_size'],
            num_triggers=1)



    sim_backdoor /= len(config.backdoors)
    z_score /= len(config.backdoors)
    
    print(f'model_save_path: {save_path}')
    print(f'sim_clean: {sim_clean}')
    print(f'sim_backdoor: {sim_backdoor}')
    print(f'z_score: {z_score}')
    
    

if __name__ == '__main__':
    config, config_path = create_parser()
    attack(config)
    
        
     