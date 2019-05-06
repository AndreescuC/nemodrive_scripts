import numpy as np
from sklearn import linear_model
from itertools import combinations


def build_coefs(Pi, Pj, ri, rj, known_z=None):
    """
    Builds the coefficients for the equation of the plane describing the intersection of two spheres (the plane on which
    the intersection circle sits). The equation is obtained by subtracting one sphere equation from another

    @:param Pi Center of the first sphere, format [x, y, z]
    @:param Pj Center of the second sphere, format [x, y, z]
    @:param ri Radius of the first sphere
    @:param rj Radius of the second sphere

    @:return y_ij, a_ij, b_ij, c_ij coefs of plane equation (y_ij = a_ij * x + b_ij * y + c_ij * z)
    """

    x_i, y_i, z_i = Pi
    x_j, y_j, z_j = Pj

    y_ij = - pow(x_i, 2) - pow(y_i, 2) - pow(z_i, 2) + pow(x_j, 2) + pow(y_j, 2) + pow(z_j, 2) - pow(rj, 2) + pow(ri, 2)
    a_ij = 2 * (x_j - x_i)
    b_ij = 2 * (y_j - y_i)
    c_ij = 2 * (z_j - z_i)

    if known_z is not None and known_z > 0:
        y_ij = y_ij - (c_ij * known_z)
        return y_ij, a_ij, b_ij

    return y_ij, a_ij, b_ij, c_ij


def solve_range_multilateration(stations, all_landmark_distances, known_zs=None):
    results = []
    for distances, known_z in zip(all_landmark_distances, known_zs):
        assert set(stations.keys()) == set(distances.keys())

        X = []
        Y = []
        for combination in combinations(list(stations), 2):
            P_i = stations[combination[0]]
            P_j = stations[combination[1]]

            dist_i = distances[combination[0]]
            dist_j = distances[combination[1]]

            if known_z is None:
                y, a, b, c = build_coefs(P_i, P_j, dist_i, dist_j, known_z=None)
                X.append([a, b, c])
            else:
                y, a, b = build_coefs(P_i, P_j, dist_i, dist_j, known_z=known_z)
                X.append([a, b])

            Y.append(y)

        clf = linear_model.LinearRegression()
        clf.fit(np.array(X), np.array(Y))

        if known_z is None:
            results.append(clf.coef_)
        else:
            results.append(np.array(clf.coef_.tolist() + [known_z]))

    return results


def optimize(results):
    return results[0]
