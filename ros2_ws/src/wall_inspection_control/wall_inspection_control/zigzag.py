def make_zigzag_goals(u_min, u_max, v_min, v_max, step):
    """
    Create inspection goals on the wall in zigzag (boustrophedon) order.

    Goals are placed on a grid with the given step. Every other row is
    reversed so that the robot never travels back across the whole row.

    :param u_min: Left edge of the scan area in wall coordinates [m].
    :param u_max: Right edge of the scan area in wall coordinates [m].
    :param v_min: Bottom edge of the scan area in wall coordinates [m].
    :param v_max: Top edge of the scan area in wall coordinates [m].
    :param step: Distance between neighboring goals [m].
    :return: List of (u, v) tuples in the order they should be visited.
    """
    # 割り算の誤差で点が1つ欠けないように丸める
    num_u = round((u_max - u_min) / step) + 1
    num_v = round((v_max - v_min) / step) + 1

    goals = []
    for j in range(num_v):
        v = v_min + j * step
        row = [(u_min + i * step, v) for i in range(num_u)]
        # 1行おきに向きを反転し、行の端から端へ戻る移動をなくす
        if j % 2 == 1:
            row.reverse()
        goals.extend(row)
    return goals
