import logging


def setup_logger(log_file='logging.log', level=logging.INFO):
    """Set up logging."""
    
    # Configure logger
    logging.basicConfig(level=level,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, 'a', 'utf-8'),
                                logging.StreamHandler()])
    
    logger = logging.getLogger()
    
    return logger

def print_and_log(logger, msg):
    # print(msg)
    try:
        logger.info(msg)
    except:
        print(msg)



def log_final_results(results: dict, args):
    """ Log all the results
    params:
    results: dict
    args: argument

    => results's format
    results = {
        'task_0': {
            'clean_acc': x,
            'poison_acc': y,
        },
        'task_1': {
            ...
        },
        'ft': {
            'clean_acc': x,
            'poison_acc': y, 
            'lr': z,
        },
        ...
    }
    """
    defense_name = {
        'nad': 'NAD',
        'pruning': 'Pruning',
        'super_ft': 'Super Finetuning'
    }

    print_and_log(args.logger, '='*50)
    print_and_log(args.logger, '='*50)
    print_and_log(args.logger, '='*50)
    print_and_log(args.logger, 'FINAL ALL RESULTS')

    for k, v in results.items():
        print_and_log(args.logger, '='*50)
        # print(v)
        clean_acc = v['clean_acc']
        poison_acc = v['poison_acc']

        if 'task' in k:
            print_and_log(args.logger, f'Result for training {k}')
            print_and_log(args.logger, f'Clean Accuracy: {clean_acc}')
            print_and_log(args.logger, f'Attack Success Rate: {poison_acc}')
        
        else:
            lr = v['lr']
            try:
                L0 = v['L0 distance']
                L1 = v['L1 distance']
                L2 = v['L2 distance']
                Linf = v['Linf distance']
            except:
                L0, L1, L2, Linf = None, None, None, None
            try:
                print_and_log(args.logger, f'Defense with {defense_name[k]}')
            except:
                print_and_log(args.logger, f'Defense with {k}')

            print_and_log(args.logger, f'Clean Accuracy: {clean_acc}')
            print_and_log(args.logger, f'Attack Success Rate: {poison_acc}')
            print_and_log(args.logger, f'Optimizer {args.ft_optimizer} with learning rate: {lr}')
            try:
                print_and_log(args.logger, f'L0 distance: \t {L0}')
                print_and_log(args.logger, f'L1 distance: \t {L1}')
                print_and_log(args.logger, f'L2 distance: \t {L2}')
                print_and_log(args.logger, f'Linf distance: \t {Linf}')
            except:
                pass



    

