run_on_dataset(){
     ./run_all_cpu.sh "EPMTitles-2.5M" "${data_version}" 0.55 1.55 "${method}"
#     ./run_all_cpu.sh "AmazonTitles-670K" "${data_version}" 0.6 2.6 "${method}"
#     ./run_all_cpu.sh "WikiSeeAlsoTitles-350K" "${data_version}" 0.55 1.5 "${method}"
#     ./run_all_cpu.sh "WikiTitles-500K" "${data_version}" 0.5 0.4 "${method}"
#     ./run_all_cpu.sh "AmazonTitles-2M" "${data_version}" 0.6 2.6 "${method}"
#     ./run_all_cpu.sh "Eurlex" "${data_version}" 0.6 2.6 "${method}"
 }

#run_on_dataset(){
#    ./run_all_cpu.sh "AmazonTitles-300K" "${data_version}" 0.6 2.6 "${method}"
#}
# run_on_dataset_large(){
#     ./run_all_cpu.sh "AmazonTitles-2M" "${data_version}" 0.6 2.6 "${method}"
# }



data_version="sparse"
# methods=("XT" "AnneXML" "ParabelOld" "PFastrexml")
# methods=("PfastreXML")
methods=("PfastreXML" "Bonsai" "AnneXML" "XT")
# methods=("PLT")
for method in "${methods[@]}"
do
    run_on_dataset
done

# for method in "${methods[@]}"
# do
#     run_on_dataset_large
# done
