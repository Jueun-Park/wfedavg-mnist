from copy import deepcopy
from itertools import permutations

def simple_weights_gen(base_index=0):
    for i in range(11):
        base_w = 0.1*i
        weights = [(1-base_w)/3 for _ in range(4)]
        weights[base_index] = base_w
        yield weights


# ???
def weights_gen(result_list, tmp_list=[], w_size=4, grid_size=8, total_weights=1, depth=0):
    if depth == w_size-1:
        tmp_list.append(total_weights)
        result_list.append(deepcopy(tmp_list))
        tmp_list.pop(-1)
        return
    for i in range(grid_size):
        now_w = i * total_weights/grid_size
        next_total_weights = total_weights - now_w
        tmp_list.append(now_w)
        weights_gen(result_list, tmp_list=tmp_list, w_size=w_size, grid_size=grid_size, total_weights=next_total_weights, depth=depth+1)
        tmp_list.pop(-1)
    return


def grid_weights_gen(w_size=4):
    a = list(range(16))
    permute = permutations(a, w_size-1)
    print(list(permute))


if __name__ == "__main__":
    grid_weights_gen()

    # result_list = []
    # weights_gen(result_list)
    # for i in range(len(result_list)):
    #     print(result_list[i])
    # print(len(result_list))
