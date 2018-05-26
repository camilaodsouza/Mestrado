#!/usr/bin/env bash

export run=train

export dataset=mnist
export model=lenet5
export exps="baseline"
python $run.py -d $dataset -lm $model -exps $exps -x 2 -e 3 #| tee artifacts/$run.$dataset.$model.$exps.txt

export dataset=cifar10
export model=vgg19bn
export exps="baseline"
python $run.py -d $dataset -lm $model -exps $exps -x 2 -e 3 #| tee artifacts/$run.$dataset.$model.$exps.txt

export dataset=cifar100
export model=vgg19bn
export exps="baseline"
python $run.py -d $dataset -lm $model -exps $exps -x 2 -e 3 #| tee artifacts/$run.$dataset.$model.$exps.txt
