#!/bin/bash

# 设定k和m的范围
k_values=(3 4 5 6 7 8 9)  # 例如k从1到5
m_values=(2 5 9) # 例如m从6到10

# 循环遍历所有k和m的组合
for k in "${k_values[@]}"
do
    for m in "${m_values[@]}"
    do
        echo "Running with k=$k and m=$m"
        java -jar esummeval_v1.2.jar /Users/huangcheng/Documents/guess2/ESBM_benchmark_v1.2 /Users/huangcheng/Documents/guess2/res_data/transe/k_${k}_m_${m}
    done
done
