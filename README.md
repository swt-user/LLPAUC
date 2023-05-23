
# NIPS_2023_LLPAUC
This is the source code for the LLPAUC. The code references Recbole (https://github.com/RUCAIBox/RecBole)

The candidate loss functions are: CCL, TP_Point_TP, TP_Point_OP, BPR, BCE, softmax

CCL:Cosin Constractive Loss

TP_Point_TP:LLPAUC Loss in our paper

TP_Point_OP:OPAUC Loss in our paper

BPR:Bayesian Personalized Ranking Loss

BCE:Binary Cross-Entropy Loss

softmax:Softmax Cross-Entropy Loss(SCE in our paper)

The candidate datasets are: adressa_clean,adressa_noise,yelp_clean,yelp_noise,amazon_book_clean,amazon_book_noise

In order to reproduce the results reported in our paper, we set the default hyper-parameters for our paper.
For example, the command to obtain the LLPAUC results for amazon_book_clean dataset is

``
python -u run_main.py --dataset=amazon_book_clean --loss=TP_Point_TP
``

