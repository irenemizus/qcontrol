#!/bin/bash
declare -a nu_L_list=("0.29297e15")
declare -a E0_list=("71.54")
declare nt=200000
declare  -a x0p_list=("-0.2" "-0.19" "-0.18" "-0.16" "-0.15" "-0.14")

for E0 in "${E0_list[@]}"
do
for nu_L in "${nu_L_list[@]}"
do
for x0p in "${x0p_list[@]}"
do
  echo "It's `date +"%T"` on the clock. Running nu_L=$nu_L, E0=${E0}, nt=${nt}, x0p=${x0p}"
  mkdir -p "output/nt${nt}/run_E${E0}_${nu_L}_x0p${x0p}"
  python3 newcheb.py --nu_L $nu_L --E0 ${E0} --nt ${nt} --x0p ${x0p} --file_abs "nt${nt}/run_E${E0}_${nu_L}_x0p${x0p}/abs_$nu_L" \
                     --file_real "nt${nt}/run_E${E0}_${nu_L}_x0p${x0p}/real_$nu_L" \
                     --file_mom "nt${nt}/run_E${E0}_${nu_L}_x0p${x0p}/mom_$nu_L" > "output/nt${nt}/run_E${E0}_${nu_L}_x0p${x0p}/out_$nu_L.txt" \
                     2> "output/nt${nt}/run_E${E0}_${nu_L}_x0p${x0p}/err_$nu_L.txt"
done
done
done
echo "It's `date +"%T"` on the clock. All done"
