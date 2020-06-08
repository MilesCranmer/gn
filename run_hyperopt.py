"""Start a hyperoptimization from a single node"""
import sys
import numpy as np
import pickle as pkl
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from quijote_gn_nv import *
import time


#Change the following code to your file
################################################################################
# TODO: Declare a folder to hold all trials objects
TRIALS_FOLDER = 'trials4'
NUMBER_TRIALS_PER_RUN = 1

def run_trial(args):
    """Evaluate the model loss using the hyperparams in args

    :args: A dictionary containing all hyperparameters
    :returns: Dict with status and loss from cross-validation

    """

    graph_data = load_graph_data(realization=0, cutoff=int(args['cutoff'])) 
    # make cutoff smaller for code to run faster (e.g., 20)
    ogn = create_graph_network(hidden=int(args['hidden']), msg_dim=int(args['msg']))
    start = time.time()
    out_loss = do_training(
	ogn, graph_data['graph'],
	total_epochs=20,#int(args['epochs']),
        batch_per_epoch=int(args['batch_per']),
        l1=args['l1'],
        weight_decay=args['l2'],
        batch=int(args['batch'])
    )
    end = time.time()

    total_time = end - start
    total_log_time = np.log10(total_time)
    successful_training = (np.min(out_loss) < 0.2)
    print(args, total_time, np.min(out_loss))

    total_loss = np.min(out_loss)
    min_loss = 0.05
    min_log_time = np.log10(10)

    combined_loss = np.sqrt(
            (total_loss - min_loss)**2 / min_loss**2 + 
            (total_log_time - min_log_time)**2 / min_log_time**2
        ) * min_loss

    return {
        'status': 'ok', # or 'fail' if inf loss
        'loss': combined_loss, # still need real loss if fail!
        'obj': total_loss,
        'time': total_time,
     }

    # if successful_training:
        # return {
            # 'status': 'ok', # or 'fail' if inf loss
            # 'loss': total_time, # still need real loss if fail!
            # '_objective': np.min(out_loss)
        # }
    # else:
        # return {
            # 'status': 'ok', # or 'fail' if inf loss
            # 'loss': 60*60*24*np.min(out_loss), # give a bad performance time if it doesn't get a good accuracy
            # '_objective': np.min(out_loss)
        # }


#TODO: Declare your hyperparameter priors here:
space = {
    'l1' : hp.loguniform('l1', np.log(1e-8), np.log(1)),
    'l2' : hp.loguniform('l2',   np.log(1e-10), np.log(1)),
    'hidden' : hp.qloguniform('hidden', np.log(10), np.log(1000+1), 1),
    'latent' : hp.qloguniform('latent', np.log(25), np.log(1000), 1),
    'msg' : hp.qloguniform('msg', np.log(1), np.log(1000), 1),
    'batch_per': hp.qloguniform('batch_per', np.log(10), np.log(30000), 1),
    'batch': hp.qloguniform('batch', np.log(10), np.log(1000), 1),
    'cutoff': hp.quniform('cutoff', 5, 50, 1)
}

################################################################################




def merge_trials(trials1, trials2_slice):
    """Merge two hyperopt trials objects

    :trials1: The primary trials object
    :trials2_slice: A slice of the trials object to be merged,
        obtained with, e.g., trials2.trials[:10]
    :returns: The merged trials object

    """
    max_tid = 0
    if len(trials1.trials) > 0:
        max_tid = max([trial['tid'] for trial in trials1.trials])

    for trial in trials2_slice:
        tid = trial['tid'] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial
        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials1.insert_trial_docs(hyperopt_trial) 
        trials1.refresh()
    return trials1

loaded_fnames = []
trials = None
# Run new hyperparameter trials until killed
while True:
    np.random.seed()

    # Load up all runs:
    import glob
    path = TRIALS_FOLDER + '/*.pkl'
    for fname in glob.glob(path):
        if fname in loaded_fnames:
            continue

        trials_obj = pkl.load(open(fname, 'rb'))
        n_trials = trials_obj['n']
        trials_obj = trials_obj['trials']
        if len(loaded_fnames) == 0: 
            trials = trials_obj
        else:
            print("Merging trials")
            trials = merge_trials(trials, trials_obj.trials[-n_trials:])

        loaded_fnames.append(fname)

    print("Loaded trials", len(loaded_fnames))
    if len(loaded_fnames) == 0:
        trials = Trials()

    n = NUMBER_TRIALS_PER_RUN
    try:
        algo = tpe.suggest
        best = fmin(run_trial,
            space=space,
            algo=algo,
            max_evals=n + len(trials.trials),
            trials=trials,
            verbose=1,
            rstate=np.random.RandomState(np.random.randint(1,10**6))
            )
    except hyperopt.exceptions.AllTrialsFailed:
        continue

    print('current best', best)
    hyperopt_trial = Trials()

    # Merge with empty trials dataset:
    save_trials = merge_trials(hyperopt_trial, trials.trials[-n:])
    new_fname = TRIALS_FOLDER + '/' + str(np.random.randint(0, sys.maxsize)) + '.pkl'
    pkl.dump({'trials': save_trials, 'n': n}, open(new_fname, 'wb'))
    loaded_fnames.append(new_fname)

