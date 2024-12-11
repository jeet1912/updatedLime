import sys
import copy
import numpy as np
import scipy as sp
import scipy.stats
from sklearn import ensemble, model_selection
import xgboost
import xgboost.sklearn
import argparse
import collections
import pickle
from load_datasets import LoadDataset
import explainers
import parzen_windows

def submodular_fn(explanations, feature_value):
    """Returns a submodular function for the given explanations."""
    z_words = set()
    for exp in explanations:
        z_words = z_words.union([x[0] for x in exp])
    normalizer = sum([feature_value[w] for w in z_words])
    
    def fnz(x):
        all_words = set()
        for doc in x:
            all_words = all_words.union([x[0] for x in explanations[doc]])
        return sum([feature_value[w] for w in all_words]) / normalizer
    
    fnz.num_items = len(explanations)
    return fnz

def greedy(submodular_fn, k, chosen=[]):
    """Greedy algorithm to maximize a submodular function."""
    chosen = copy.deepcopy(chosen)
    all_items = list(range(submodular_fn.num_items))
    current_value = 0
    while len(chosen) < k:
        best_gain = 0
        best_item = all_items[0]
        for i in all_items:
            gain = submodular_fn(chosen + [i]) - current_value
            if gain > best_gain:
                best_gain = gain
                best_item = i
        chosen.append(best_item)
        all_items.remove(best_item)
        current_value += best_gain 
    return chosen

def submodular_pick(pickled_map, explainer, B, use_explanation_weights=False, alternate=False):
    """Selects explanations based on submodular optimization."""
    def get_function(exps):
        feature_value = collections.defaultdict(float)
        for exp in exps:
            for f, v in exp:
                if not use_explanation_weights:
                    v = 1
                feature_value[f] += np.abs(v)
        for f in feature_value:
            feature_value[f] = np.sqrt(feature_value[f])
        submodular = submodular_fn(exps, feature_value)
        return submodular

    if explainer in ['parzen', 'lime']:
        exps1 = [x[0] for x in pickled_map['exps1'][explainer]]
        exps2 = [x[0] for x in pickled_map['exps2'][explainer]]
    else:
        exps1 = pickled_map['exps1'][explainer]
        exps2 = pickled_map['exps2'][explainer]

    fn1 = get_function(exps1)
    fn2 = get_function(exps2)
    
    if not alternate:
        return greedy(fn1, B), greedy(fn2, B)
    else:
        ret = []
        for i in range(B):
            fn = fn1 if i % 2 == 0 else fn2
            ret = greedy(fn, i + 1, ret)
        return ret

def all_pick(pickled_map, explainer, B):
    """Picks all explanations."""
    list_ = list(range(len(pickled_map['exps1'][explainer])))
    return list_, list_

def random_pick(pickled_map, explainer, B):
    """Randomly picks explanations."""
    list_ = np.random.choice(range(len(pickled_map['exps1'][explainer])), B, replace=False)
    return list_, list_

def find_untrustworthy(explainer, exps, instances, untrustworthy):
    """Finds untrustworthy features in explanations."""
    found = set()
    for i in instances:
        if explainer in ['lime', 'parzen']:
            exp, _ = exps[i]
        else:
            exp = exps[i]
        found = found.union([x[0] for x in exp if x[0] in untrustworthy])
    return found

def tally_mistrust(explainer, exps, predict_probas, untrustworthy):
    """Counts the number of mispredictions due to untrustworthy features."""
    trust_fn = lambda prev, curr: (prev > 0.5 and curr > 0.5) or (prev <= 0.5 and curr <= 0.5)
    mistrust = 0
    for i in range(len(exps)):
        if explainer in ['lime', 'parzen']:
            exp, mean = exps[i]
            prev_tot = predict_probas[i] if explainer == 'lime' else mean
            tot = prev_tot - sum([x[1] for x in exp if x[0] in untrustworthy])
            if not trust_fn(tot, prev_tot):
                mistrust += 1
        else:
            exp = exps[i]
            if len([x[0] for x in exp if x[0] in untrustworthy]) > 0:
                mistrust += 1
    return mistrust

def main():
    parser = argparse.ArgumentParser(description='Evaluate explanations for classifiers')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
    parser.add_argument('--output_folder', '-o', type=str, required=True, help='Output folder')
    parser.add_argument('--num_features', '-k', type=int, required=True, help='Number of features')
    parser.add_argument('--pick', '-p', type=str, default='all', help='Method to pick explanations: all, submodular, or random')
    parser.add_argument('--num_instances', '-n', type=int, default=1, help='Number of instances to look at')
    parser.add_argument('--num_rounds', '-r', type=int, default=10, help='Number of rounds for random pick')
    args = parser.parse_args()

    dataset = args.dataset
    got_right = lambda test1, test2, mistrust1, mistrust2: mistrust1 < mistrust2 if test1 > test2 else mistrust1 > mistrust2
    names = ['lime', 'parzen', 'random', 'greedy']
    num_exps = 0
    B = args.num_instances
    rounds = 1 if args.pick != 'random' else args.num_rounds

    # Define pick function based on argument
    pick_functions = {
        'all': all_pick,
        'submodular': lambda a,b,c : submodular_pick(a,b,c, use_explanation_weights=True),
        'random': random_pick
    }
    pick_function = pick_functions.get(args.pick, all_pick)

    for r in range(rounds):
        for filez in sorted(glob.glob(os.path.join(args.output_folder, f'comparing_{args.dataset}*')))[:800]:
            num_exps += 1
            with open(filez, 'rb') as f:
                pickled_map = pickle.load(f)
            predict_probas = pickled_map['predict_probas1']
            predict_probas2 = pickled_map['predict_probas2']
            test1, test2 = pickled_map['test1'], pickled_map['test2']
            
            # Process for each explainer
            for explainer in names:
                instances1, instances2 = pick_function(pickled_map, explainer, B)
                untrustworthy = pickled_map['untrustworthy']
                mistrust1 = tally_mistrust(explainer, pickled_map['exps1'][explainer], predict_probas, untrustworthy)
                mistrust2 = tally_mistrust(explainer, pickled_map['exps2'][explainer], predict_probas2, untrustworthy)
                
                if got_right(test1, test2, mistrust1, mistrust2):
                    right[explainer].append(1)
                else:
                    right[explainer].append(0)

    # Output results
    for explainer in names:
        if right[explainer]:
            print(f"{explainer}: {sum(right[explainer])}/{len(right[explainer])}")

if __name__ == "__main__":
    main()
