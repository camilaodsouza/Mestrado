#!/usr/bin/env bash

export run=train

#export dataset=mnist
#for model in lenet5
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
#done

export dataset=cifar10
for model in alexnet vgg19bn resnet101 squeezenet squeezemobnet mobilenet
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
done

export dataset=cifar100
for model in alexnet vgg19bn resnet101 squeezenet squeezemobnet mobilenet
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
done

#export dataset=imagenet
#for model in alexnet vgg19_bn resnet101 squeezenet1_0 squeezenet1_1
#do
#  export exps="baseline"
#  python $run.py -d $dataset -rm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=imagenet
#for model in squeezemobnet1_0 squeezemobnet1_1 mobilenet
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
#done
