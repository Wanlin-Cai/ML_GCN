先使用prepare_data_mpnn生成数据graph.bin

在utils.py中指定了in_feats=9/12

训练
python regression_train.py -c 5000reorg.csv -sc smiles -s random
python regression_train.py -c 5000reorg.csv -sc smiles -s random -ne 128
(需要的文件为model_configures/,analysis.py,hyper.py,mol_to_graph.py,regression_train.py,utils.py)

预测
使用inference_prepare_data中修改的文件


在regression_train.py加上如下信息，得到分类情况
    test_smiles = [i[0] for i in test_set]
    val_smiles = [i[0] for i in val_set]
    train_smiles = [i[0] for i in train_set]
    np.savetxt('test_smiles.txt',np.array(test_smiles),fmt='%-200s')
    np.savetxt('val_smiles.txt',np.array(val_smiles),fmt='%-200s')
    np.savetxt('train_smiles.txt',np.array(train_smiles),fmt='%-200s')