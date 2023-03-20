import numpy as np
from pygemma import pygemma_model

# TODO: Implement GEMMA model call
# See https://github.com/genetics-statistics/GEMMA/blob/master/src/lmm.cpp#L2208
def pygemma():
    UtW = np.dot(U.T W)
    UtY = np.dot(U.T, Y)

    pygemma_model.CalcLambda('L', eval, UtW, &UtY_col.vector, cPar.l_min, cPar.l_max,
                   cPar.n_region, cPar.l_mle_null, cPar.logl_mle_H0)
    
    pygemma_model.CalcLmmVgVeBeta(eval, UtW, &UtY_col.vector, cPar.l_mle_null,
                        cPar.vg_mle_null, cPar.ve_mle_null, &beta.vector,
                        &se_beta.vector)
    
    

