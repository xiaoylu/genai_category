# Diffusion

## `diffusion_mnist.ipynb`

Steps:
* download mnist data
* build a UNet using pytorch
* add extra time mask embedding to inform which step in diffusion sampling 
* add extra condition embedding to guide the generate (a digit from 0 - 9)
* train to predict the synthetic noises 
* sampling using the model to generate images of the digit 

## `diffusion_mnist_controlnet.ipynb`

On top of `diffusion_mnist.ipynb`,
* UNet: remove the condition embedding (ie. FCC taking input of digit 0 - 9)
* UNet: use the upper half of image as the guide (idea in ControlNet)  
* Train without guide first
* Finetune with guide while freezing the remaining parameters
* Test using mnist_test data 

