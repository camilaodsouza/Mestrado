#!/usr/bin/env bash

mkdir -p images/val
tar -xvf patch_images.tar
tar -xvf ILSVRC2010_devkit-1.0.tar.gz
python create_val_script.py
tar -xvf ILSVRC2010_images_val.tar -C images
mv patch_images/val/* images/val
bash val_script.sh
