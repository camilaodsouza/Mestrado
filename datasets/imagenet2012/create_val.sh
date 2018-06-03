#!/usr/bin/env bash

#mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
#wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

mkdir -p images/val
tar -xvf ILSVRC2012_devkit_t12.tar.gz
python create_val_script.py
tar -xvf ILSVRC2012_img_val.tar -C images/val
bash val_script.sh
