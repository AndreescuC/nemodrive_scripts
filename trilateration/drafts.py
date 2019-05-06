

# def solve_range_multilateration(base_world_coordinates, all_landmark_distances):
#     results = {}
#     for landmark, distances in enumerate(all_landmark_distances):
#         assert set(base_world_coordinates.keys()) == set(distances.keys())
#
#         local_results = []
#         for combination in combinations(list(base_world_coordinates), 3):
#             P1 = base_world_coordinates[combination[0]]
#             P2 = base_world_coordinates[combination[1]]
#             P3 = base_world_coordinates[combination[2]]
#
#             distA = distances[combination[0]]
#             distB = distances[combination[1]]
#             distC = distances[combination[2]]
#
#             local_results.append(solve_trilateration(P1, P2, P3, distA, distB, distC))
#
#         results[landmark] = optimize(results)


# def solve_trilateration(P1, P2, P3, r1, r2, r3):
#     temp1 = P2 - P1
#     e_x = temp1 / np.linalg.norm(temp1)
#
#     temp2 = P3 - P1
#     i = np.dot(e_x, temp2)
#
#     temp3 = temp2 - i * e_x
#     e_y = temp3 / np.linalg.norm(temp3)
#
#     e_z = np.cross(e_x, e_y)
#
#     d = np.linalg.norm(P2 - P1)
#     j = np.dot(e_y, temp2)
#
#     x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
#     y = (r1 * r1 - r3 * r3 - 2 * i * x + i * i + j * j) / (2 * j)
#     temp4 = r1 * r1 - x * x - y * y
#     if temp4 < 0:
#         raise Exception("The three spheres do not intersect!");
#     z = np.sqrt(temp4)
#     p_12_a = P1 + x * e_x + y * e_y + z * e_z
#     p_12_b = P1 + x * e_x + y * e_y - z * e_z
#     return x, y, z


# def solve_trilateration(P1, P2, P3, DistA, DistB, DistC):
#     ex = (P2 - P1) / (np.linalg.norm(P2 - P1))
#     i = np.dot(ex, P3 - P1)
#     ey = (P3 - P1 - i * ex) / (np.linalg.norm(P3 - P1 - i * ex))
#     d = np.linalg.norm(P2 - P1)
#     j = np.dot(ey, P3 - P1)
#
#     x = (pow(DistA, 2) - pow(DistB, 2) + pow(d, 2)) / (2 * d)
#     y = ((pow(DistA, 2) - pow(DistC, 2) + pow(i, 2) + pow(j, 2)) / (2 * j)) - ((i / j) * x)
#
#     z_squared = pow(DistA, 2) - pow(x, 2) - pow(y, 2)
#     if z_squared >= 0:
#         return x, y, np.sqrt(z_squared)
#
#     raise Exception("The three spheres do not intersect!");