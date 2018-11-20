#!/usr/bin/env bash

export run=train

#export dataset=cifar10
#for model in squeezenet_ squeezemobnet_ mobilenet_ mobilenetv2_ alexnet_
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps -x 5 | tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=cifar100
#for model in alexnet_ squeezenet_ squeezemobnet_ mobilenet_ mobilenetv2_
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps -x 5 #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

export dataset=imagenet2012
for model in squeezemobnet #squeezenet1_1 mobilenet mobilenetv2 alexnet
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps -x 1 | tee artifacts/$run.$dataset.$model.$exps.txt
done
