import pandas as pd
# from pgmpy.estimators import BDeuScore, K2Score, BicScore, HillClimbSearch

from bnlearn import dag2adjmat
import pybnesian

def hillclimbsearch(df: pd.DataFrame, score='bic', verbose=3):
    # TODO black_list, white_list
    # TODO max_indegree=max_indegree, tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter,
    #    fixed_edges=fixed_edges, show_progress=False

    out = {}

    model = pybnesian.hc(df, bn_type=pybnesian.DiscreteBNType(), )

    # scoring_method = _SetScoringType(df, score, verbose=verbose)
    # model = HillClimbSearch(df)
    #
    # best_model = model.estimate(scoring_method=scoring_method)

    # Store
    out['model'] = best_model
    out['model_edges'] = list(out['model'].edges())
    out['adjmat'] = dag2adjmat(out['model'])

    if verbose >= 4:
        print(best_model.edges())

    return out


def _SetScoringType(df, scoretype, verbose=3):
    if verbose >= 3: print('[bnlearn] >Set scoring type at [%s]' % (scoretype))

    scoring_method = None

    if scoretype == 'bic':
        scoring_method = BicScore(df)
    elif scoretype == 'k2':
        scoring_method = K2Score(df)
    elif scoretype == 'bdeu':
        scoring_method = BDeuScore(df, equivalent_sample_size=5)

    return scoring_method
