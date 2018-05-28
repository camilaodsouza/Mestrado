#!/usr/bin/env bash

export run=train

#export dataset=mnist
#for model in lenet5
#do
#  export exps="nmc~5"
#  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=cifar10
#for model in vgg19bn
#do
#  export exps="nmc~5"
#  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
#done

export dataset=cifar100
for model in vgg19bn
do
  export exps="seed~1234+nmc~50"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
  export exps="seed~12345+nmc~50"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
  export exps="seed~123456+nmc~50"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
done
