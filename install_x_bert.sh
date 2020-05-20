cd X-BERT
conda create -n xbert-env python=3.6
conda activate xbert-env
conda install scikit-learn
conda install pytorch=0.4.1 cuda90 -c pytorch
pip install urllib3==1.24
pip install pytorch-pretrained-bert==0.6.2
pip install allennlp==0.8.4
pip install -e .
conda deactivate
cd -