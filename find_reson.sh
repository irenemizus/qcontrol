#!/bin/bash
declare -a nu_L_list=("2.89e14" "2.90e14" "2.91e14" "2.915e14" "2.92e14" "2.925e14" "2.9275e14" "2.93e14" "2.935e14" "2.9325e14" "2.935e14" "2.94e14" "2.945e14" "2.95e14" "2.97e14")
declare -a E0_list=("300")
declare nt=100000
for E0 in "${E0_list[@]}"
do
for nu_L in "${nu_L_list[@]}"
do
  echo "It's `date +"%T"` on the clock. Running nu_L=$nu_L, E0=${E0}, nt=${nt}"
  mkdir -p "output/nt${nt}/run_E${E0}_$nu_L"
  python3 newcheb.py --nu_L $nu_L --E0 ${E0} --nt ${nt} --file_abs "nt${nt}/run_E${E0}_$nu_L/abs_$nu_L" \
                     --file_real "nt${nt}/run_E${E0}_$nu_L/real_$nu_L" \
                     --file_mom "nt${nt}/run_E${E0}_$nu_L/mom_$nu_L" > "output/nt${nt}/run_E${E0}_$nu_L/out_$nu_L.txt" \
                     2> "output/nt${nt}/run_E${E0}_$nu_L/err_$nu_L.txt"
done
done
echo "It's `date +"%T"` on the clock. All done"
