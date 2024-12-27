import numpy as np
from typing import Dict, List, Set, Tuple
from new_gd_ds import GradDataStruct

if __name__ == "__main__":
    # 3 users: u1, u2, u3
    # 4 items: i1, i2, i3, i4
    # dimension = 2
    # u1 rate for i1, i2
    # u2 rate for i2, i3
    # u3 rate for i1

    gds = GradDataStruct(dimension=2)

    gds.add_rating("u1", "i1")
    gds.add_rating("u1", "i2")
    gds.add_rating("u2", "i2")
    gds.add_rating("u2", "i3")
    gds.add_rating("u3", "i1")
    gds.finalize()

    print("M:", gds.M)
    print("MI:", gds.MI)
    print("MJ:", gds.MJ)

    A = np.array([
        [1.0, 1.0],  # corresponds to (u1,i1)
        [2.0, 2.0],  # corresponds to (u1,i2)
        [3.0, 3.0],  # corresponds to (u2,i1)
        [4.0, 4.0],  # corresponds to (u2,i3)
        [5.0, 5.0],  # corresponds to (u3,i1)
    ])

    agg_u_res = gds.agg_u(A)
    print("\nagg_u result (sums in MI order):")
    for user_id, sum_vec in zip(gds.MI, agg_u_res):
        print(f"{user_id}: {sum_vec}")
        
    agg_v_res = gds.agg_v(A)
    print("\nagg_v result (sums in MJ order):")
    for item_id, sum_vec in zip(gds.MJ, agg_v_res):
        print(f"{item_id}: {sum_vec}")
    
     # Test reconstitutions
    rec_u_res = gds.rec_u(agg_u_res)
    print("\nrec_u result (reconstruct vectors for each M pair):")
    for (i, j), vec in zip(gds.M, rec_u_res):
        print(f"({i},{j}): {vec}")
        
    rec_v_res = gds.rec_v(agg_v_res)
    print("\nrec_v result (reconstruct vectors for each M pair):")
    for (i, j), vec in zip(gds.M, rec_v_res):
        print(f"({i},{j}): {vec}")
