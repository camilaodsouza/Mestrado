#!/usr/bin/env bash

export run=train

export dataset=cifar10
for model in squeezemobnet_ #alexnet_ squeezenet_ squeezemobnet_ mobilenet_ mobilenetv2_
do
  export exps="bsd~1230_bsd~12340_bsd~123450_bsd~1234560"
  python $run.py -d $dataset -lm $model -exps $exps | tee artifacts/$run.$dataset.$model.$exps.txt
done

#export dataset=cifar10
#for model in alexnet_ squeezenet_ squeezemobnet_ mobilenet_ mobilenetv2_
#do
#  export exps="bsd~1230" #_benchnodeter+bsd~12340_benchnodeter+bsd~123450_benchnodeter+bsd~1234560"
#  python $run.py -d $dataset -lm $model -exps $exps -x 10 -e 150 | tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=cifar100
#for model in alexnet_ squeezenet_ squeezemobnet_ mobilenet_ mobilenetv2_
#do
#  export exps="bsd~1230" #_benchnodeter+bsd~12340_benchnodeter+bsd~123450_benchnodeter+bsd~1234560"
#  python $run.py -d $dataset -lm $model -exps $exps -x 10 -e 150 | tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=imagenet2012
#for model in alexnet squeezenet1_1 squeezemobnet1_1 mobilenet mobilenetv2
#do
#  export exps="bsd~1230" #_benchnodeter+bsd~12340_benchnodeter+bsd~123450_benchnodeter+bsd~1234560"
#  python $run.py -d $dataset -lm $model -exps $exps -x 3 -e 100 | tee artifacts/$run.$dataset.$model.$exps.txt
#done
