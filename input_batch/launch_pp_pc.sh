echo "Running post_processing.sh"
python ../post_processing.py ../../outputs/unitary_transform/Hamil_BH_2lvls_Jx/results_from_cluster/bu_25.01.23/output_ut_Jx_600/250-9200fs/ ../../outputs/unitary_transform/Hamil_BH_2lvls_Jx/results_from_cluster/bu_25.01.23/output_ut_Jx_600/250-9200fs/  > ../batch_jsons_out/out_pp.txt 2>../batch_jsons_out/err_pp

