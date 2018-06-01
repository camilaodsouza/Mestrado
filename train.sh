#!/usr/bin/env bash

export run=train

#export dataset=mnist
#for model in lenet5
#do
#  export exps="baseline+nmc~5"
#  python $run.py -d $dataset -lm $model -exps $exps -e 3 -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

export dataset=cifar10
for model in resnet500 #shufflenetg2 #resnet50 #vgg19bn #alexnet
do
  export exps="baseline+nmc~5"
  python $run.py -d $dataset -lm $model -exps $exps -e 3 -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
done

export dataset=cifar100
for model in mobilenetv2 #resnet50 #vgg19bn #alexnet
do
  export exps="baseline+nmc~50"
  python $run.py -d $dataset -lm $model -exps $exps -e 3 -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
done
