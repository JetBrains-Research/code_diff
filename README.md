# Code Diffusion

Implementation of Diffusion LM model and its extensions aimed for source code processing. Repositories used are [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) and [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), original papers: ["Diffusion-LM Improves Controllable Text Generation"](https://arxiv.org/pdf/2205.14217.pdf).

Execute
`python3 -m pip install -r requirements.txt`
to install necessary dependencies

Execute 
`python3 -m scripts.text.train_diffusion_lm`
to train LM

Execute 
`python3 -m scripts.text.train_controlling_classifier
to train syntax tree controlling classifier
