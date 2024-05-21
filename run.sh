#!/bin/bash
# Scripts to get the results

k_values=(3 4 5 6 7 8 9)  
m_values=(2 5 9)


for k in "${k_values[@]}"
do
    for m in "${m_values[@]}"
    do
        echo "Running with k=$k and m=$m"
        java -jar esummeval_v1.2.jar /Documents/guess2/ESBM_benchmark_v1.2 /Documents/guess2/res_data/transe/k_${k}_m_${m}
    done
done
