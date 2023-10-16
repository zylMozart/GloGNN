import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--het_threshold', type=float, default=1.0) ##
    parser.add_argument('--temp', type=float, default=1.0) ##
    parser.add_argument('--lambda', type=float, default=100.0) ##
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight for frobenius norm on Z.')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Weight for MLP results kept')
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Weight for nodes feature kept')
    parser.add_argument('--norm_layers', type=int, default=2,
                        help='Number of groupnorm layers')
    parser.add_argument('--dataset', type=str, default='pubmed',
                        help='Name of dataset')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Name of dataset')
    parser.add_argument('--split', type=int, default=0,
                        help='Split part of dataset')
    parser.add_argument('--early_stopping', type=int, default=40,
                        help='Early stopping')
    parser.add_argument('--model', type=str, default='gcn',
                        help='Model name ')
    parser.add_argument('--orders', type=int, default=2,
                        help='Number of adj orders in norm layer')
    parser.add_argument('--orders_func_id', type=int, default=3,
                        help='Sum function of adj orders in norm layer, ids \in [1, 2, 3]')
    parser.add_argument('--norm_func_id', type=int, default=2,
                        help='Function of norm layer, ids \in [1, 2]')

    return parser