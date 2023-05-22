tp_hyper={
    ('adressa_clean','CCL'):
        {
            'neg_margin': -0.5
        },
    ('adressa_clean','TP_Point_OP'):
        {
            'beta': 0.5
        },
    ('adressa_clean','TP_Point_TP'):
        {
            'alpha':0.7,
            'beta':0.1
        },
    ('adressa_noise', 'CCL'):
        {
            'neg_margin': -0.5
        },
    ('adressa_noise', 'TP_Point_OP'):
        {
            'beta': 0.5
        },
    ('adressa_noise', 'TP_Point_TP'):
        {
            'alpha': 0.7,
            'beta': 0.1
        },
    ('yelp_clean', 'CCL'):
        {
            'neg_margin': -0.2
        },
    ('yelp_clean', 'TP_Point_OP'):
        {
            'beta': 0.02
        },
    ('yelp_clean', 'TP_Point_TP'):
        {
            'alpha': 0.9,
            'beta': 0.01
        },
    ('yelp_noise', 'CCL'):
        {
            'neg_margin': -0.2
        },
    ('yelp_noise', 'TP_Point_OP'):
        {
            'beta': 0.02
        },
    ('yelp_noise', 'TP_Point_TP'):
        {
            'alpha': 1.1,
            'beta': 0.01
        },
    ('amazon_book_clean', 'CCL'):
        {
            'neg_margin': 0.0
        },
    ('amazon_book_clean', 'TP_Point_OP'):
        {
            'beta': 0.02
        },
    ('amazon_book_clean', 'TP_Point_TP'):
        {
            'alpha': 0.9,
            'beta': 0.01
        },
    ('amazon_book_noise', 'CCL'):
        {
            'neg_margin': 0.0
        },
    ('amazon_book_noise', 'TP_Point_OP'):
        {
            'beta': 0.02
        },
    ('amazon_book_noise', 'TP_Point_TP'):
        {
            'alpha': 0.9,
            'beta': 0.01
        }
}
def get_hyper(dataset,loss):
    if (dataset,loss) in tp_hyper:
        return tp_hyper[(dataset,loss)]
    else:
        return {}
