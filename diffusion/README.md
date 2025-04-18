# Diffusion

## `diffusion_mnist.ipynb`

Steps:
* download mnist data
* build a UNet using pytorch
* add extra time mask embedding to inform which step in diffusion sampling 
* add extra condition embedding to guide the generate (a digit from 0 - 9)
* train to predict the synthetic noises 
* sampling using the model to generate images of the digit 

