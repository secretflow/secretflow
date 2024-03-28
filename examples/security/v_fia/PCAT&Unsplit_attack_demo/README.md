# Introduction
This folder is intended to demonstrate an attack demonstration of split learning. 
The attack process is specifically referred to the paper 《PCAT: Functionality and Data Stealing from Split Learning
by Pseudo-Client Attack》and《UnSplit Data-Oblivious Model Inversion, Model Stealing, and Label Inference Atatck Against Split Learning》,
the code was developed based on Secretflow and optimized for our needs.


# PCAT&Unsplit_attack_demo
- PCAT_attack_demo
  - model
    - alice
    - bob
  - _utils_attack.py
  - attacks.py
  - datasets.py
  - model_attack.py
  - ndarray_attack.py
  - sl_base_pcat.py
  - sl_model_pcat.py
  - split_pcat.py
  - README.md

# Use Tips
- PCAT

  If you want to run this program, you can simply run 'split_pcat.py' with optional arguments '-h or --help' and '-v or --value' , 
  - 'wt' means stealing the model during training 
  - 'at' means stealing after training.  
  -  example:`python split_pcat.py -v wt`

  **Result**: At the end of the program, three images will be obtained, 
which are the original image and two restored images. 
The first restored image is  the smash data  that restored by decoder 
model which is obtained from the attacker model. 
The second restored image is restored by decoder model, 
and the smash data is obtained from the victim model.  


- Unsplit
  
  example:`python split_Unsplit.py`

  **Result**: Show the reconstructed image

