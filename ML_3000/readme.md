��ʹ��prepare_data_mpnn��������graph.bin

��utils.py��ָ����in_feats=9/12

ѵ��
python regression_train.py -c 5000reorg.csv -sc smiles -s random
python regression_train.py -c 5000reorg.csv -sc smiles -s random -ne 128
(��Ҫ���ļ�Ϊmodel_configures/,analysis.py,hyper.py,mol_to_graph.py,regression_train.py,utils.py)

Ԥ��
ʹ��inference_prepare_data���޸ĵ��ļ�


��regression_train.py����������Ϣ���õ��������
    test_smiles = [i[0] for i in test_set]
    val_smiles = [i[0] for i in val_set]
    train_smiles = [i[0] for i in train_set]
    np.savetxt('test_smiles.txt',np.array(test_smiles),fmt='%-200s')
    np.savetxt('val_smiles.txt',np.array(val_smiles),fmt='%-200s')
    np.savetxt('train_smiles.txt',np.array(train_smiles),fmt='%-200s')