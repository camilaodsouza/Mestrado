#!/usr/bin/env bash

export run=train

export dataset=mnist
export model=lenet5
export exps="nmc~5+rt~ne+rv~0.0"
export exps=$exps"_nmc~5+rt~l2+rv~0.0001_nmc~5+rt~l2+rv~0.001_nmc~5+rt~l2+rv~0.01"
export exps=$exps"_nmc~5+rt~ne+rv~0.1_nmc~5+rt~ne+rv~0.3_nmc~5+rt~ne+rv~0.5"
python $run.py -d $dataset -lm $model -exps $exps -x 10 -tr -el | tee artifacts/$run.$dataset.$model.$exps.txt

export dataset=cifar10
export model=vgg19bn
export exps="nmc~5+rt~ne+rv~0.0"
export exps=$exps"_nmc~5+rt~l2+rv~0.0001_nmc~5+rt~l2+rv~0.001_nmc~5+rt~l2+rv~0.01"
export exps=$exps"_nmc~5+rt~ne+rv~0.1_nmc~5+rt~ne+rv~0.3_nmc~5+rt~ne+rv~0.5"
python $run.py -d $dataset -lm $model -exps $exps -x 10 -tr -el | tee artifacts/$run.$dataset.$model.$exps.txt

export dataset=cifar100
export model=vgg19bn
export exps="nmc~50+rt~ne+rv~0.0"
export exps=$exps"_nmc~50+rt~l2+rv~0.0001_nmc~50+rt~l2+rv~0.001_nmc~50+rt~l2+rv~0.01"
export exps=$exps"_nmc~50+rt~ne+rv~0.1_nmc~50+rt~ne+rv~0.3_nmc~50+rt~ne+rv~0.5"
python $run.py -d $dataset -lm $model -exps $exps -x 10 -tr -el | tee artifacts/$run.$dataset.$model.$exps.txt
