#!/usr/bin/env bash

export run=train

export dataset=cifar10
for model in alexnet vgg19bn resnet101 mobilenet squeezenet squeezemobnet
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps -e 3 -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
done

export dataset=cifar100
for model in alexnet vgg19bn resnet101 mobilenet squeezenet squeezemobnet
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps -e 3 -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
done

#export dataset=imagenet
#for model in squeezenet1_0 squeezenet1_1
#do
#  export exps="baseline"
#  python $run.py -d $dataset -rm $model -exps $exps -e 3 -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=imagenet
#for model in squeezemobnet1_0 squeezemobnet1_1
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps -e 3 -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
#done
