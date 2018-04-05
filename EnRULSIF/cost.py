import numpy as np

def normal_mean(data, variance):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    cd = [0.0]
    cd.extend(np.cumsum(data))

    cd2 = [0.0]
    cd2.extend(np.cumsum(data*data))
    start = 0
    end = len(cd2)-1
    cd2_diff = cd2[end] - cd2[start]
    cd_diff = pow(cd[end] - cd[start], 2)
    i_diff = end - start
    diff = cd2_diff - cd_diff/i_diff
    cost = diff/variance

    return cost