#!/bin/bash

for k in {0..9}
do
  for i in {0..5}
  do
     python train_ner_batch.py $i $k
  done
done



