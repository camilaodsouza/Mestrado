#!/usr/bin/env bash

export run=train

export dataset=cifar10
for model in squeezemobnet_ alexnet_ squeezenet_ mobilenet_ mobilenetv2_
do
  export exps="bsd~1230" #_benchnodeter+bsd~12340_benchnodeter+bsd~123450_benchnodeter+bsd~1234560"
  python $run.py -d $dataset -lm $model -exps $exps -x 1 -e 2 | tee artifacts/$run.$dataset.$model.$exps.txt
done

export dataset=cifar100
for model in squeezemobnet_ alexnet_ squeezenet_ mobilenet_ mobilenetv2_
do
  export exps="bsd~1230" #_benchnodeter+bsd~12340_benchnodeter+bsd~123450_benchnodeter+bsd~1234560"
  python $run.py -d $dataset -lm $model -exps $exps -x 1 -e 2 | tee artifacts/$run.$dataset.$model.$exps.txt
done

export dataset=imagenet2012
for model in squeezemobnet alexnet squeezenet #mobilenet mobilenetv2
do
  export exps="bsd~1230" #_benchnodeter+bsd~12340_benchnodeter+bsd~123450_benchnodeter+bsd~1234560"
  python $run.py -d $dataset -lm $model -exps $exps -x 1 -e 2 | tee artifacts/$run.$dataset.$model.$exps.txt
done
