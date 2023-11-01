在服务器中使用prepare_data_for_interference.py生成数据graph_interference.bin

然后使用修改版的下文件
python regression_inference.py -f 5000reorg.csv -sc smiles

需要把utils.py文件及, regression_results/model.pth,configure.json文件夹拷贝过来