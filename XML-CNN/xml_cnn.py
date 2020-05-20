import sys
import _pickle as cPickle

root = sys.argv[1]
text_file_trn = root+'/train_raw_full.txt'
labl_file_trn = root+'/train_X_Y.txt'
text_file_tst = root+'/test_raw_full.txt'
labl_file_tst = root+'/test_X_Y.txt'

trn_txt = open(text_file_trn, 'r', encoding='latin')
trn_lbl = open(labl_file_trn, 'r')

instances, labels = map(int, trn_lbl.readline().split(' '))
trn_data = []
split = 'train'
keep_train_ind = [(instances, labels)]
for inst in range(instances):
    inst_dict = {}
    inst_dict['text'] = trn_txt.readline().strip()
    inst_dict['split'] = split

    inst_dict['num_words'] = len(inst_dict['text'].split(' '))
    inst_dict['catgy'] = list(map(lambda x: x.split(
        ":")[0], trn_lbl.readline().strip().split(' ')))
    if inst_dict['num_words'] > 0 and len(inst_dict['catgy']) > 0:
        inst_dict['Id'] = str(inst)
        trn_data.append(inst_dict)
        keep_train_ind.append(inst)
    print("[%d/%d]" % (inst, instances), end='\r')

tst_txt = open(text_file_tst, 'r', encoding='latin')
tst_lbl = open(labl_file_tst, 'r')

instances, labels = map(int, tst_lbl.readline().split(' '))
tst_data = []
split = 'test'
keep_test_ind = [(instances, labels)]
for inst in range(instances):
    inst_dict = {}
    inst_dict['text'] = tst_txt.readline().strip()
    inst_dict['split'] = split
    inst_dict['num_words'] = len(inst_dict['text'].split(' '))
    inst_dict['catgy'] = list(map(lambda x: x.split(
        ":")[0], tst_lbl.readline().strip().split(' ')))
    if inst_dict['num_words'] > 0 and len(inst_dict['catgy']) > 0:
        inst_dict['Id'] = str(inst)
        tst_data.append(inst_dict)
        keep_test_ind.append(inst)
    print("[%d/%d]" % (inst, instances), end='\r')

vocab = {"Useless": 0.0}
catgy = {}
for i in range(labels):
    catgy[str(i)] = i

dataset = open(sys.argv[1]+'/xml_cnn.p', 'wb')
cPickle.dump([trn_data, tst_data, vocab, catgy, keep_train_ind,
              keep_test_ind], dataset, protocol=2)
dataset.close()
