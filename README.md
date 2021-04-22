## Train SOTA Architectures on benchmark Data-sets with `PyTorch`


Dependencies (other than common ones):
- `diffeomorphism` https://github.com/pcsl-epfl/diffeomorphism
- Experiments are run using `grid` https://github.com/mariogeiger/grid/tree/master/grid


The list of _parameters_ includes:
 - Dataset (see below) - `ptr` indicates the train-set size.
 - Architecture (see below)
 - Optimizer (`sgd`, `adam`)
 - lr and lr scheduler (`cosineannealing`, `none`)
 - loss function (`crossentropy` for multi-class, `hinge` for one class)
 - training in feature or lazy regime with `alpha`-trick (`featlazy` to 1 and vary `alpha`)
 - ...
 

Example:

        python main.py --epochs 200 --save_best_net 1 --save_dynamics 0 --diffeo 0 --batch_size 32 --net:str 'EfficientNetB0' --dataset:str 'cifar10' --seed_init 0 --ptr 1024




| Datasets             |
| ----------------- |
| mnist
| fashionmnist
| cifar10



**Models** impelementations are based on [github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
The list includes:

| Models             |
| ----------------- |
| [VGG16](https://arxiv.org/abs/1409.1556)
| [ResNet18](https://arxiv.org/abs/1512.03385)
| [ResNet50](https://arxiv.org/abs/1512.03385)
| [ResNet101](https://arxiv.org/abs/1512.03385)
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678) 
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678) 
| [MobileNetV2](https://arxiv.org/abs/1801.04381)
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)
| [SimpleDLA](https://arxiv.org/abs/1707.064) 
| [DenseNet121](https://arxiv.org/abs/1608.06993) 
| [PreActResNet18](https://arxiv.org/abs/1603.05027) 
| [DPN92](https://arxiv.org/abs/1707.01629) 
| [DLA](https://arxiv.org/abs/1707.064) 


## Measuring nets' function stability 

The folder `results` contains the tools for computing a function stability toward (i) random transformations of the input (`G_f`), (ii) diffeomorphisms (`D_f`) and (iii) the relative stability `R_f` as defined in "[paper]". 

Also, details about the trainings performed for "[paper]" are provided.
