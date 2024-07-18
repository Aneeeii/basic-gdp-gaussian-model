import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def gaussian(year):
    data = pd.read_excel('reformatted.xlsx', header=0)

    X = data[["year"]]
    y = data[["high", "medium", "low"]]

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr_models = {}
    for col in y.columns:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(X, y[col])
        gpr_models[col] = gpr

    def interpolate_box_gpr(z_value):
        predictions = [gpr_models[col].predict([[z_value]], return_std = True) for col in y.columns]
        return predictions

    predicted_coords_gpr = interpolate_box_gpr(year)
    return predicted_coords_gpr[0][0], predicted_coords_gpr[1][0], predicted_coords_gpr[2][0]


def run_this():
    inp = input("Input year here(quit or q to stop): ")
    while inp != "quit" or inp != "q":
        result = gaussian(int(inp))
        print(result)
        inp = input("Input year here(quit or q to stop): ")


if __name__ == "__main__":
    run_this()