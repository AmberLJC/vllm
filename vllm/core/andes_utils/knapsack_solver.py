from typing import Optional, Literal, List, Deque, Tuple, Callable
from collections import deque
import time 

from vllm.sequence import SequenceGroup
 

class KnapSack():
    """
    A class representing a knapsack solver for request scheduling.

    Args:
    block_size (int): The size of each block.
    total_available_blocks (int): The total number of available blocks.
    solver (Literal['greedy', 'dp'], optional): The solver algorithm to use. Defaults to 'greedy'.
    """
    def __init__(self, 
                 block_size: int, 
                 total_available_blocks: int, 
                 solver: Literal['greedy', 'dp'] = 'greedy',
                 ) -> None:
        self.solver = solver 
        # TODO: move this to somewhere else
        self.delta_t = 50
        self.block_size = block_size
        self.total_available_blocks = total_available_blocks  
        self.cnt = 0
 
        # TODO: consider compute

    def pick_requests(self, running: Deque[SequenceGroup], waiting: Deque[SequenceGroup], swapped: Deque[SequenceGroup]) -> List[SequenceGroup]:
        """
        Picks the requests among the running, waiting, and swapped sequence list.

        Args:
        running (Deque[SequenceGroup]): The running sequence list.
        waiting (Deque[SequenceGroup]): The waiting sequence list.
        swapped (Deque[SequenceGroup]): The swapped sequence list.

        Returns:
        List[SequenceGroup]: The picked sequence groups.
        """
        def get_context_block_list(requests: List[SequenceGroup]) -> List[int]:
            return [max(1, round(r.get_len() / self.block_size + .5)) for r in requests]

        running, waiting, swapped = list(running), list(waiting), list(swapped)
        now = time.monotonic()

        max_batchsize = len(running) + len(waiting) + len(swapped)
        context_block_list = get_context_block_list(running + waiting + swapped)
        value_list = [r.get_value(now, 0.1, self.delta_t, True) for r in running]
        value_list += [r.get_value(now, 0.1, self.delta_t, False) for r in waiting + swapped]
        max_value, best_plan = self.knap_sack(self.total_available_blocks, max_batchsize, context_block_list, value_list)
        if max_value <= 0:
            return running
        return [(running + waiting + swapped)[i] for i in best_plan]

    def schedule_requests(self, budget: int, running: Deque[SequenceGroup], 
                          waiting: Deque[SequenceGroup], 
                          swapped: Deque[SequenceGroup], 
                          utilization: float, 
                          latency_function: Callable) -> Tuple[Deque[SequenceGroup], Deque[SequenceGroup], Deque[SequenceGroup]]: 
        # TODO: consider budget
        self.cnt += 1
        if utilization < 0.9:
            return deque(waiting), deque(swapped), deque()
        elif self.cnt % 5 != 0:
            return deque(), deque(), deque()
        
        new_running = self.pick_requests(running, waiting, swapped)
        seq_to_admit = [r for r in new_running if r in waiting]
        seq_to_swap_in = [r for r in new_running if r in swapped]
        seq_to_evict = [r for r in running if r not in new_running]
 
        return deque(seq_to_admit), deque(seq_to_swap_in), deque(seq_to_evict)

    def knap_sack(self, W: int, item_cap: int, wt: list, val: list ) -> int: 
        """
        Solves the knapsack problem using either the greedy or dynamic programming algorithm.

        Args:
        W (int): The maximum weight the knapsack can carry.
        item_cap (int): The maximum number of items to pick.
        wt (list): The list of weights of the items.
        val (list): The list of values of the items.

        Returns:
        int: The total value of the picked items.
        """
        def _greedy_knapsack(capacity: int, item_cap: int, weights: list, values: list) -> tuple:
            """
            Greedy algorithm for the fractional knapsack problem.

            Args:
            capacity (int): Maximum weight the knapsack can carry.
            weights (list): List of weights of the items.
            values (list): List of values of the items.

            Returns:
            tuple: Total value of the picked items and the list of indices of the picked items.
            """
            if sum(weights) <= capacity:
                # pick top item_cap items with highest value
                picked_items = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:item_cap]
                return sum([values[i] for i in picked_items]), picked_items
 
            # Calculate value-to-weight ratio and sort items by this ratio in descending order
            items = sorted([(v / w, w, v, i) for i, (v, w) in enumerate(zip(values, weights))], reverse=True)
            current_weight = 0
            current_value = 0
            picked_items = []

            for ratio, weight, value, index in items:
                if current_weight + weight <= capacity and len(picked_items) < item_cap:
                    current_weight += weight
                    current_value += value
                    picked_items.append(index)
                else:
                    break

            return current_value, picked_items
        
        def _dp_knapsack(M: int, B: int, weights: List[int], values: List[int]) -> Tuple[int, List[int]]: 
            """
            Dynamic programming algorithm for the knapsack problem.

            Args:
            M (int): Maximum weight the knapsack can carry.
            B (int): Maximum number of items to pick.
            weights (List[int]): List of item weights.
            values (List[int]): List of item values.

            Returns:
            Tuple[int, List[int]]: Total value of the picked items and the list of indices of the picked items.
            """
            N = len(values) 
            dp = [[[-float('inf') for _ in range(M+1)] for _ in range(B+1)] for _ in range(N+1)]
            choice = [[[0 for _ in range(M+1)] for _ in range(B+1)] for _ in range(N+1)]
            dp[0][0][0] = 0  

            # Main DP loop
            for i in range(1, N+1):
                for b in range(min(i, B)+1):  # +1 because range is exclusive
                    for m in range(M+1):
                        # Case 1: Do not select the i-th request
                        if dp[i][b][m] < dp[i-1][b][m]:
                            dp[i][b][m] = dp[i-1][b][m]
                            choice[i][b][m] = 0

                        # Case 2: Select the i-th request, if it does not exceed the constraints
                        if b > 0 and m + weights[i-1] <= M and dp[i][b][m + weights[i-1]] < dp[i-1][b-1][m] + values[i-1]:
                            dp[i][b][m + weights[i-1]] = dp[i-1][b-1][m] + values[i-1]
                            choice[i][b][m + weights[i-1]] = 1

            # Find the optimal solution and corresponding memory
            max_value, current_m = max((value, index) for index, value in enumerate(dp[N][B]))

            # Backtrack to find the decision array x_i
            x = [0] * (N+1)  # Initialize decision array with zeros
            current_b = B
            for i in range(N, 0, -1):
                x[i] = choice[i][current_b][current_m]
            if x[i] == 1:
                current_m -= weights[i-1]
                current_b -= 1
     
            picked_items = [i for i in range(1, N+1) if x[i] == 1] 
            return max_value, picked_items

        assert len(wt) == len(val)
        if self.solver == 'greedy':
            return _greedy_knapsack(W, item_cap, wt, val)
        elif self.solver == 'dp':
            return _dp_knapsack(W, item_cap, wt, val)
        else:
            raise ValueError(f"Solver {self.solver} is not supported")
    