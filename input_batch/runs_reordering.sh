#!/bin/bash
for i in {1..20}
do
  let "j = $i + $1"
  mv ../output_ut/run$i ../output_ut/run$j
done
