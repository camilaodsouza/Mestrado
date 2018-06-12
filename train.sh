#!/usr/bin/env bash

export run=train

#export dataset=mnist
#for model in lenet5_mnist
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
#done

export dataset=cifar10
for model in squeezemobnet mobilenet #alexnet squeezenet
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
done

export dataset=cifar100
for model in alexnet squeezenet squeezemobnet mobilenet
do
  export exps="baseline"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
done

#export dataset=imagenet
#for model in squeezemobnet1_0_imagenet squeezemobnet1_1_imagenet mobilenet_imagenet
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps -x 1 #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=imagenet
#for model in alexnet squeezenet1_0 squeezenet1_1
#do
#  export exps="baseline"
#  python $run.py -d $dataset -rm $model -exps $exps #| tee artifacts/$run.$dataset.$model.$exps.txt
#done
