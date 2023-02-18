import bnlearn as bn
# import pybnesian
# import bnlearn.pybnesian as bns
#
# # %%
# # Load asia DAG
df = bn.import_example(data='asia')
df = df.astype('float64')
# # Structure learning of sampled dataset
# # model = bn.structure_learning.fit(df)
#
#
# # model = pybnesian.hc(df, bn_type=pybnesian.DiscreteBNType(), score='bic', operators=['arcs'], verbose=True)
# hc = pybnesian.GreedyHillClimbing()
# operators = pybnesian.ArcOperatorSet()
# score=pybnesian.BIC(df)
# start=pybnesian.DiscreteBN(nodes=list(df.columns))
#
# model2 = hc.estimate(operators=operators, score=score, start=start, verbose=1)
#
#
#
#
#
# # model2 = bns.structure_learning.hillclimbsearch(df)
#
#
#
# # Make plot
# G = bn.plot(model2)
#


import numpy as np

np.random.seed(1)

import pandas as pd

DATA_SIZE = 100
a_array = np.random.normal(3, np.sqrt(0.5), size=DATA_SIZE)

c_array = -4.2 - 1.2 * a_array + np.random.normal(0, np.sqrt(0.75), size=DATA_SIZE)
d_array = 3 + 1.2 * c_array + np.random.normal(0, np.sqrt(0.5), size=DATA_SIZE)
e_array = np.random.normal(0, 1, size=DATA_SIZE)
df = pd.DataFrame({'a': a_array, 'c': c_array, 'd': d_array, 'e': e_array})


from pybnesian import hc, GaussianNetworkType, DiscreteBNType
# learned = hc(df, bn_type=GaussianNetworkType())
learned = hc(df, bn_type=DiscreteBNType(), score='bic', operators=['arcs'])

learned.num_arcs()



df.dtypes