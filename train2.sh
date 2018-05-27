#!/usr/bin/env bash

export run=train2

#export dataset=mnist
#for model in lenet5
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps #| tee artifacts2/$run.$dataset.$model.$exps.txt
#done

export dataset=cifar10
for model in vgg19bn
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps #| tee artifacts2/$run.$dataset.$model.$exps.txt
done

#export dataset=cifar100
#for model in vgg19bn
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps #| tee artifacts2/$run.$dataset.$model.$exps.txt
#done
