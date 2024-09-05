This repository contains the data and code for the paper named "Ying-Zhang_2.pdf".

Install
Install dependencies with pip install -r requirements.txt

Datasets
Datasets used in this study are selected from UCI and KEEL dataset repositories, as shown in the "data" directory.

Code
The core code consists of three parts, as shown in the "code/code" directory:

1. Data Generation:
   - Script: 
	-EnhanceGAN_generate.py

2. MCMC Sampling:
   - Scripts:
     - EnhanceGAN_MCMC.py
     - en_repgan.py
     - calibration.py
     - MLP.py

3. Classification:
   - Script: Enhance_classifier.py

Execution Commands:
To reproduce the results published in the paper, execute the following commands sequentially:
```
python EnhanceGAN_generate.py
python EnhanceGAN_MCMC.py
python EnhanceGAN_classifier.py
```