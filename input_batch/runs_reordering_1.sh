#!/bin/bash
for i in {1..25}
do
  let "j = $i + $1"
  mv ../output1_ut/run$i ../output1_ut/run$j
done
