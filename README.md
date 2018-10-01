# fire
efficient modules for pytorch to make it more easier.
Abstract Modules
* Trainer
* Estimator
* Logger

## Usage
```
import torch
import torch.nn.functional as F
import torchvision
import fire
import models

model = models.CNN()
loss_func = F.nll_loss
train_data = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

args = fire.Trainer.get_args()
args_dict = vars(args)
trainer = fire.Trainer(**args_dict)
trainer.fit(model, train_data, test_data,loss_func)
```

## Requirements
* pytorch
* torchvision
* tqdm
* comet_ml(optional)

## Install
```
$ git clone https://github.com/mzntaka0/fire.git
$ python setup.py install
```




