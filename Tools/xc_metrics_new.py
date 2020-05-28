"""
    Compute evaluation statistics.
"""
import scipy.sparse as sp
import numpy as np
from xclib.utils.sparse import topk, binarize, retain_topk
from collections import OrderedDict
import csv
import pdb
#from utils import *

__author__ = 'X'

def myarr( vec ):
    return np.asarray( vec ).flatten()

def jaccard_similarity(pred_0, pred_1, y=None): 
    """Jaccard similary b/w two different predictions matrices
    Args:
    pred_0: csr_matrix
        prediction for algorithm 0
    pred_1: csr_matrix
        prediction for algorithm 1
    y: csr_matrix or None
        true labels
    """
    def _correct_only(pred, y):
        pred = pred.multiply(y)
        pred.eliminate_zeros()
        return pred

    def _safe_divide(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.true_divide(a, b)
            out[out == np.inf] = 0
            return np.nan_to_num(out)

    if y is not None:
        pred_0 = _correct_only(pred_0, y)
        pred_1 = _correct_only(pred_1, y)

    pred_0, pred_1 = binarize(pred_0), binarize(pred_1)
    intersection = np.array(pred_0.multiply(pred_1).sum(axis=1)).ravel()
    union = np.array(binarize(pred_0 + pred_1).sum(axis=1)).ravel()
    return np.mean(_safe_divide(intersection, union))


def recall(predicted_labels, true_labels, k=5):
    """Compute recall@k
    Args:
    predicted_labels: csr_matrix
        predicted labels
    true_labels: csr_matrix
        true_labels
    k: int, default=5
        keep only top-k predictions
    """
    predicted_labels = retain_topk(predicted_labels, k)    #write k=k
    denom = np.sum(true_labels, axis=1)
    rc = binarize(true_labels.multiply(predicted_labels))
    rc = np.sum(rc, axis=1)/(denom+1e-5)
    return np.mean(rc)*100


def format(*args, decimal_points='%0.2f'):
    out = []
    for vals in args:
        out.append(','.join(list(map(lambda x: decimal_points % (x*100), vals))))
    return '\n'.join(out)


def compute_inv_propesity(labels, A, B):
    num_instances, _ = labels.shape
    freqs = np.ravel(np.sum(labels>0, axis=0))
    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(freqs+B, -A)
    return np.ravel(wts)


class Metrices(object):
    def __init__(self, true_labels, inv_propensity_scores=None, remove_invalid=False, batch_size=20):
        """
            Args:
                true_labels: csr_matrix: true labels with shape (num_instances, num_labels)
                remove_invalid: boolean: remove samples without any true label
        """
        self.true_labels = true_labels
        self.num_instances, self.num_labels = true_labels.shape
        self.remove_invalid = remove_invalid
        self.valid_idx = None
        if self.remove_invalid:
            samples = np.sum(self.true_labels, axis=1)
            self.valid_idx = np.arange(self.num_instances).reshape(-1, 1)[samples > 0]
            self.true_labels = self.true_labels[self.valid_idx]
            self.num_instances = self.valid_idx.size
        "inserting dummpy index"
        self.true_labels_padded = sp.hstack(
            [self.true_labels, sp.csr_matrix(np.zeros((self.num_instances, 1)))]).tocsr()
        #print(max(self.true_labels_padded))
        self.ndcg_denominator = np.cumsum(
            1/np.log2(np.arange(1, self.num_labels+1)+1)).reshape(-1, 1)
        self.labels_documents = np.ravel(np.array(np.sum(self.true_labels, axis=1), np.int32))
        self.labels_documents[self.labels_documents == 0] = 1
        self.inv_propensity_scores = None
        self.batch_size = batch_size
        if inv_propensity_scores is not None:
            self.inv_propensity_scores = np.hstack(
                [inv_propensity_scores, np.zeros((1))])
            assert(self.inv_propensity_scores.size == self.true_labels_padded.shape[1])

    def _rank_sparse(self, X, K):
        """
            Args:
                X: csr_matrix: sparse score matrix with shape (num_instances, num_labels)
                K: int: Top-k values to rank
            Returns:
                predicted: np.ndarray: (num_instances, K) top-k ranks
        """
        total = X.shape[0]
        labels = X.shape[1]

        predicted = np.full((total, K), labels)
        for i, x in enumerate(X):
            index = x.__dict__['indices']
            data = x.__dict__['data']
            idx = np.argsort(-data)[0:K]
            predicted[i, :idx.shape[0]] = index[idx]
        return predicted

    def print_metrics(self,metrics, mets, w):
        #print("mets are ",metrics.keys())
        for met in mets:
            if met in metrics:
                metvec = metrics[met]
                #print("%s:\t1: %.2f\t3: %.2f\t5: %.2f" %( met, metvec[0]*100, metvec[2]*100, metvec[4]*100 ))
                val=[metvec[0]*100, metvec[2]*100, metvec[4]*100]        
                w.writerow([met]+list(map(str,val)))          

    def eval(self, predicted_labels, w, mets = ['P','PSP','CTR','N','PSN'],K=5):
        """
            Args:
                predicted_labels: csr_matrix: predicted labels with shape (num_instances, num_labels)
                K: int: compute values from 1-5
        """
        if self.valid_idx is not None:
            predicted_labels = predicted_labels[self.valid_idx]
        assert predicted_labels.shape == self.true_labels.shape
        
        score_mat = retain_topk(predicted_labels,k=K)      
        predicted_labels = topk(predicted_labels, K, self.num_labels, -100)        
        metrics = OrderedDict()
        if 'P' in mets:
            prec = self.precision(predicted_labels, K)
            #prec = self.my_precision(score_mat, K)
            metrics['P'] = prec

        if 'RP' in mets:
            rp = self.my_RP(score_mat, K)
            metrics['RP'] = rp

        if 'N' in mets:
            ndcg = self.nDCG(predicted_labels, K)            
            metrics['N'] = ndcg
        
        if 'CTR' in mets:
            ctr = self.my_ctr(score_mat, K)
            metrics['CTR'] = ctr

        if self.inv_propensity_scores is not None:
            wt_true_mat = self._rank_sparse(self.true_labels.dot(sp.spdiags(
                self.inv_propensity_scores[:-1], diags=0, m=self.num_labels, n=self.num_labels)), K)
            if 'PSP' in mets:
                PSprecision = self.PSprecision(predicted_labels, K) / self.PSprecision(wt_true_mat, K)
                metrics['PSP'] = PSprecision
            if 'PSN' in mets:
                PSnDCG = self.PSnDCG(predicted_labels, K) / self.PSnDCG(wt_true_mat, K)                
                metrics['PSN'] = PSnDCG
            #return [prec, ndcg, PSprecision, PSnDCG]
            #return [prec, ndcg]
        self.print_metrics(metrics, mets, w)
        return metrics

    def precision(self, predicted_labels, K):
        """
            Compute precision for 1-K
        """
        p = np.zeros((1, K))
        total_samples = self.true_labels.shape[0]
        ids = np.arange(total_samples).reshape(-1, 1)
        p = np.sum(self.true_labels_padded[ids, predicted_labels]>0, axis=0)
        p = p*1.0/(total_samples)
        p = np.cumsum(p)/(np.arange(K)+1)
        return np.ravel(p)    

    def my_precision(self, score_mat, K):   #not P@k
        V = np.empty( K )
        for i in range(K,0,-1):
            score_mat = retain_topk(score_mat,k=i)
            mul = (score_mat>0).multiply(self.true_labels)
            V[i-1] = 1.*np.sum(mul)/(i*self.true_labels.shape[0])
        #pdb.set_trace()
        return V

    def my_ctr(self, score_mat, K):
        Ptop = self.my_precision(score_mat, K)
        Pbottom = self.my_precision(self.true_labels, K)
        P = np.divide(Ptop, Pbottom)
        return P

    def my_RP(self, score_mat, K):
        V = np.empty( K )
        for i in range(K,0,-1):
            #pdb.set_trace()
            score_mat = retain_topk(score_mat, k=i)
            top = myarr(np.sum((score_mat>0).multiply(self.true_labels>0), axis=1))
            labs = myarr(np.sum(self.true_labels>0, axis=1) )
            bottom = myarr([ labs[ind] if labs[ind]<i else i for ind in range(len(labs))])
            V[i-1] = np.sum(np.divide(top, bottom))/self.true_labels.shape[0]
        #pdb.set_trace()
        return V

    def my_nDCG(self, predicted_labels, K):
        """
            Compute nDCG for 1-K
        """
        ndcg = np.zeros((1, K))
        total_samples = self.true_labels.shape[0]
        ids = np.arange(total_samples).reshape(-1, 1)
        pdb.set_trace()
        dcg = self.true_labels_padded[ids, predicted_labels].power(2) - 1
        dcg = dcg/(np.log2(np.arange(1, K+1)+1)).reshape(1, -1)
        
        dcg = np.cumsum(dcg, axis=1)
        denominator = self.ndcg_denominator[self.labels_documents-1]
        for k in range(K):
            temp = denominator.copy()
            temp[denominator > self.ndcg_denominator[k]] = self.ndcg_denominator[k]
            temp = np.power(temp, -1.0)
            for batch in np.array_split(np.arange(total_samples), self.batch_size):
                dcg[batch, k] = np.ravel(np.multiply(dcg[batch, k], temp[batch]))
            ndcg[0, k] = np.mean(dcg[:, k])
            del temp
        return np.ravel(ndcg)

    def nDCG(self, predicted_labels, K):
        """
            Compute nDCG for 1-K
        """
        ndcg = np.zeros((1, K))
        total_samples = self.true_labels.shape[0]
        ids = np.arange(total_samples).reshape(-1, 1)
        dcg = self.true_labels_padded[ids, predicted_labels] /(
            np.log2(np.arange(1, K+1)+1)).reshape(1, -1)
        dcg = np.cumsum(dcg, axis=1)
        denominator = self.ndcg_denominator[self.labels_documents-1]
        for k in range(K):
            temp = denominator.copy()
            temp[denominator > self.ndcg_denominator[k]] = self.ndcg_denominator[k]
            temp = np.power(temp, -1.0)
            for batch in np.array_split(np.arange(total_samples), self.batch_size):
                dcg[batch, k] = np.ravel(np.multiply(dcg[batch, k], temp[batch]))
            ndcg[0, k] = np.mean(dcg[:, k])
            del temp
        return np.ravel(ndcg)

    def PSnDCG(self, predicted_labels, K):
        """
            Compute PSnDCG for 1-K
        """
        psndcg = np.zeros((1, K))
        total_samples = self.true_labels.shape[0]
        ids = np.arange(total_samples).reshape(-1, 1)

        ps_dcg = self.true_labels_padded[ids, predicted_labels].toarray(
        )*self.inv_propensity_scores[predicted_labels]/np.log2(np.arange(1, K+1)+1).reshape(1, -1)
        
        ps_dcg = np.cumsum(ps_dcg, axis=1)
        denominator = self.ndcg_denominator[self.labels_documents-1]
        
        for k in range(K):
            temp = denominator.copy()
            temp[denominator > self.ndcg_denominator[k]] = self.ndcg_denominator[k]
            temp = np.power(temp, -1.0)
            for batch in np.array_split(np.arange(total_samples), self.batch_size):
                ps_dcg[batch, k] = ps_dcg[batch, k]*temp[batch, 0]
            psndcg[0, k] = np.mean(ps_dcg[:, k])
            del temp
        return np.ravel(psndcg)

    def PSprecision(self, predicted_labels, K):
            """
                Compute PSprecision for 1-K
            """
            psp = np.zeros((1, K))
            total_samples = self.true_labels.shape[0]
            ids = np.arange(total_samples).reshape(-1, 1)
            _p = (self.true_labels_padded[ids, predicted_labels]>0).toarray(
            )*self.inv_propensity_scores[predicted_labels]
            psp = np.sum(_p, axis=0)
            psp = psp*1.0/(total_samples)
            psp = np.cumsum(psp)/(np.arange(K)+1)
            return np.ravel(psp)


def get_Y(trn,type):
    if type=="seen":
        f = myarr(np.sum(trn>0,axis = 0))
        ys = np.where(f>0)[0]
        return ys
    if type == "new":
        f = myarr(np.sum(trn>0,axis = 0))
        ys = np.where(f==0)[0]
        #print("new labels are",len(ys))
        return ys

    if type == "frequent":
        f = myarr(np.sum(trn>0,axis = 0))
        ys = np.where(f>=50)[0]
        print("frequent labels are", len(ys))
        return ys
    if type == "few":
        f = myarr(np.sum(trn>0,axis = 0))
        ys = np.where(abs(f-25)<25)[0]
        print("few labels are ", len(ys))
        return ys
    return

def eurlex_do_metrics(trn, tst, score_mat, inv_propen, file):
    mets = ['P','PSP','N','RP']
    f = open( file + ".csv", "w")
    w = csv.writer(f)  
    f.close()
    w = csv.writer(open( file + ".csv", "a"))  
    w.writerow(['All labels'])  
    print("all labels are",tst.shape[1])  
    #print("All labels\n")
    f = myarr(np.sum(tst>0,axis = 1))
    dps = np.where(f>0)[0]
    print("all dps",len(dps))
    tst = tst[dps,:]
    score_mat = score_mat[dps,:]
    acc = Metrices(tst, inv_propensity_scores=inv_propen, remove_invalid=False, batch_size=50000)
    ans = acc.eval(score_mat, w, mets = mets, K=5)
    return
    #print("Seen labels\n")
    w.writerow(['Frequent labels'])
    old_labs = get_Y(trn,"frequent")
    inv_propen_old = inv_propen[old_labs]
    tst_old = tst[:,old_labs]
    f = myarr(np.sum(tst_old>0,axis = 1))
    dps = np.where(f>0)[0]
    print("frequent dps",len(dps))
    tst_old = tst_old[dps,:]
    score_mat_old = score_mat[dps,:]
    score_mat_old = score_mat_old[:,old_labs]
    acc = Metrices(tst_old, inv_propensity_scores=inv_propen_old, remove_invalid=False, batch_size=50000)
    ans = acc.eval(score_mat_old, w, mets = mets, K=5)

    w.writerow(['Few labels'])
    new_labs = get_Y(trn,"few")
    inv_propen_new = inv_propen[new_labs]
    tst_new = tst[:,new_labs]
    f = myarr(np.sum(tst_new>0,axis = 1))
    dps = np.where(f>0)[0] 
    print("few dps",len(dps))   
    tst_new = tst_new[dps,:]
    score_mat_new = score_mat[dps,:]
    score_mat_new = score_mat_new[:,new_labs]
    acc = Metrices(tst_new, inv_propensity_scores=inv_propen_new, remove_invalid=False, batch_size=50000)
    ans = acc.eval(score_mat_new, w, mets = mets, K=5)

    w.writerow(['New labels'])
    new_labs = get_Y(trn,"new")
    inv_propen_new = inv_propen[new_labs]
    tst_new = tst[:,new_labs]
    f = myarr(np.sum(tst_new>0,axis = 1))
    dps = np.where(f>0)[0]
    print("zero dps",len(dps))
    if(len(dps)==0):
        w.writerow(['No tst pts with new labels'])
    else:
        tst_new = tst_new[dps,:]
        score_mat_new = score_mat[dps,:]
        score_mat_new = score_mat_new[:,new_labs]
        acc = Metrices(tst_new, inv_propensity_scores=inv_propen_new, remove_invalid=False, batch_size=50000)
        ans = acc.eval(score_mat_new, w, mets = mets, K=5)

def do_metrics(trn, tst, score_mat, inv_propen, file):
    f = open( file + ".csv", "w")
    w = csv.writer(f)  
    f.close()
    w = csv.writer(open( file + ".csv", "a"))  
    w.writerow(['All labels'])    
    #print("All labels\n")
    f = myarr(np.sum(tst>0,axis = 1))
    dps = np.where(f>0)[0]
    tst = tst[dps,:]
    score_mat = score_mat[dps,:]
    acc = Metrices(tst, inv_propensity_scores=inv_propen, remove_invalid=False, batch_size=50000)
    ans = acc.eval(score_mat, w, mets = ['P','PSP', 'N', 'RP', 'CTR'], K=5)

    #print("Seen labels\n")
    w.writerow(['Seen labels'])
    old_labs = get_Y(trn,"seen")
    inv_propen_old = inv_propen[old_labs]
    tst_old = tst[:,old_labs]
    f = myarr(np.sum(tst_old>0,axis = 1))
    dps = np.where(f>0)[0]
    tst_old = tst_old[dps,:]
    score_mat_old = score_mat[dps,:]
    score_mat_old = score_mat_old[:,old_labs]
    acc = Metrices(tst_old, inv_propensity_scores=inv_propen_old, remove_invalid=False, batch_size=50000)
    ans = acc.eval(score_mat_old, w, mets = ['P','PSP', 'N', 'RP', 'CTR'], K=5)

    #print("Unseen labels\n")
    w.writerow(['Unseen labels'])
    new_labs = get_Y(trn,"new")
    inv_propen_new = inv_propen[new_labs]
    tst_new = tst[:,new_labs]
    f = myarr(np.sum(tst_new>0,axis = 1))
    dps = np.where(f>0)[0]
    if(len(dps)==0):
        w.writerow(['No tst pts with new labels'])
    else:
        tst_new = tst_new[dps,:]
        score_mat_new = score_mat[dps,:]
        score_mat_new = score_mat_new[:,new_labs]
        acc = Metrices(tst_new, inv_propensity_scores=inv_propen_new, remove_invalid=False, batch_size=50000)
        ans = acc.eval(score_mat_new, w, mets = ['P','PSP', 'N', 'RP', 'CTR'], K=5)

    #print("Labelwise\n")
    w.writerow(['Labelwise'])
    trn = trn.transpose().tocsr()    
    score_mat = score_mat.transpose().tocsr()    
    tst = tst.transpose().tocsr()
    #inv_propen = compute_inv_propesity(trn, A, B)
    
    f = myarr(np.sum(tst>0,axis = 1))
    ys = np.where(f>0)[0]
    tst = tst[ys,:]
    score_mat = score_mat[ys,:]     
    acc = Metrices(tst, remove_invalid=False, batch_size=50000)
    ans = acc.eval(score_mat, w, mets = ['P', 'N', 'RP', 'CTR'], K=5)
