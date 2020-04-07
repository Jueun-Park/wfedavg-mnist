from copy import deepcopy
from itertools import combinations_with_replacement

def simple_weights_gen(base_index=0):
    for i in range(11):
        base_w = 0.1*i
        weights = [(1-base_w)/3 for _ in range(4)]
        weights[base_index] = base_w
        yield weights


# ???
def divided_weights_gen(result_list, tmp_list=[], w_size=4, grid_size=8, total_weights=1, depth=0):
    if depth == w_size-1:
        tmp_list.append(total_weights)
        result_list.append(deepcopy(tmp_list))
        tmp_list.pop(-1)
        return
    for i in range(grid_size):
        now_w = i * total_weights/grid_size
        next_total_weights = total_weights - now_w
        tmp_list.append(now_w)
        divided_weights_gen(result_list, tmp_list=tmp_list, w_size=w_size, grid_size=grid_size, total_weights=next_total_weights, depth=depth+1)
        tmp_list.pop(-1)
    return


def grid_weights_gen(w_size=4, grid_size=16):
    a = list(range(grid_size+1))
    combi = combinations_with_replacement(a, w_size-1)
    block_size = 1 / grid_size
    result = []
    for c in combi:
        assigned = 0
        temp = []
        for bar in c:
            num = bar - assigned
            w = num * block_size
            temp.append(w)
            assigned += num
        num = grid_size - assigned
        w = num * block_size
        temp.append(w)
        result.append(temp)
            
    return result



if __name__ == "__main__":
    li = grid_weights_gen()
    for i in range(len(li)):
        print(li[i])
    print(len(li))

    # result_list = []
    # divided_weights_gen(result_list)
    # for i in range(len(result_list)):
    #     print(result_list[i])
    # print(len(result_list))
