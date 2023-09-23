import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="TransGraph Models",
        usage="Waiting to add"
    )

    parser.add_argument('--use_init', default=0, type=int)
    parser.add_argument('--eval_type', default='bottom', type=str)
    parser.add_argument('-c', '--cuda', action='store_true', help='use GPU')
    parser.add_argument('-model', default="ATransE", type=str)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument('--do_valid', default=True, type=bool)
    parser.add_argument('--do_test', default=1, type=int)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--data_path', type=str, default='e_ori_kge/data/FB15k-237')

    parser.add_argument('--neg_ratio', default=1, type=int)
    parser.add_argument('-o', '--one', default=0, type=int)
    parser.add_argument('-d', '--hidden_dim', default=100, type=int)
    parser.add_argument('--lmbda', default=0, type=float)
    parser.add_argument('-g', '--gamma', default=9.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-r', '--regularization', default=0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=5e-6, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    # parser.add_argument('-init', '--init_checkpoint', default="e_ori_kge/models/ATransE1_FB15k-237", type=str)
    parser.add_argument('-save', '--save_path', default="models", type=str)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--scheduler_step', default=25000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=50, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)
