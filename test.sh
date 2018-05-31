#!/usr/bin/env bash

export run=test

#export dataset=mnist
#for model in lenet5
#do
#  export exps="bs~1234+nmc~5_bs~12345+nmc~5_bs~123456+nmc~5"
##  export exps="nmc~5"
##  #export exps=$exps"_nmc~5+rt~l2+rv~0.0001_nmc~5+rt~l2+rv~0.001_nmc~5+rt~l2+rv~0.01"
##  #export exps=$exps"_nmc~5+rt~ne+rv~0.1_nmc~5+rt~ne+rv~0.3_nmc~5+rt~ne+rv~0.5"
#  python $run.py -d $dataset -lm $model -exps $exps -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=cifar10
#for model in vgg19bn
#do
#  export exps="nmc~5"
#  #export exps=$exps"_nmc~5+rt~l2+rv~0.0001_nmc~5+rt~l2+rv~0.001_nmc~5+rt~l2+rv~0.01"
#  #export exps=$exps"_nmc~5+rt~ne+rv~0.1_nmc~5+rt~ne+rv~0.3_nmc~5+rt~ne+rv~0.5"
#  python $run.py -d $dataset -lm $model -exps $exps #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

export dataset=cifar100
for model in vgg19bn
do
  export exps="ebs~1230+nmc~50_ebs~12340+nmc~50_ebs~123450+nmc~50"
  #export exps="nmc~50"
  #export exps=$exps"_nmc~50+rt~l2+rv~0.0001_nmc~50+rt~l2+rv~0.001_nmc~50+rt~l2+rv~0.01"
  #export exps=$exps"_nmc~50+rt~ne+rv~0.1_nmc~50+rt~ne+rv~0.3_nmc~50+rt~ne+rv~0.5"
  python $run.py -d $dataset -lm $model -exps $exps -x 2 #| tee artifacts/$run.$dataset.$model.$exps.txt
done
