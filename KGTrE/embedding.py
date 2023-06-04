from .model import *
from .sampling import *
import os
import pickle
def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

def embedding(args):
    embed_dim = args.dim
    metric_dim = 32
    batch_size = 512
    learning_rate = args.lr
    l2 = args.l2
    neg_num = 3
    total_batch = 10000
    display_batch = 200
    if args.dataset == 'city':
        with open("data_city.pkl", 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            fo.close()
    if args.dataset == 'dblp':
        with open("data/dblp/data_dblp.pkl", 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            fo.close()

    model = KGTrE(data, embed_dim, metric_dim, batch_size, learning_rate, l2, args)

    u_s, u_d = sampling_paths(data, neg_num, numwalks=2, size=2)
    avg_loss = 0

    for i in range(total_batch):
        sdx = (i * batch_size) % len(u_s)
        edx = ((i + 1) * batch_size) % len(u_s)
        if edx > sdx:
            u_si = u_s[sdx:edx]
        else:
            u_si = u_s[sdx:] + u_s[0:edx]
        sdx = (i * batch_size) % len(u_d)
        edx = ((i + 1) * batch_size) % len(u_d)
        if edx > sdx:
            u_di = u_d[sdx:edx]
        else:
            u_di = u_d[sdx:] + u_d[:edx]

        loss = model.train_line(u_si, u_di)
        avg_loss += loss / display_batch

    args.embeddings = model.cal_embed()
    args.relation = model.cal_relation()

    return args

