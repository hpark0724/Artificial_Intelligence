import heapq
import copy as cp

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    # initialize the distance
    distance = 0
    # loop through each tile and its index in the from_state
    for index, tile in enumerate(from_state):
        # skip the tile with 0
        if tile == 0:
            continue
        # calculate the column and row of the from_state (3 by 3)
        col_from, row_from = index % 3, index // 3

        # the index of the to_state tile
        index = to_state.index(tile)
        # calculate the column and row of the to_state (3 by 3)
        col_to, row_to = index % 3, index // 3

        # calculate its the manhatten distance 
        distance += abs(col_from - col_to) + abs(row_from - row_to)
    # return the total manhatten distance
    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))



def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """

    # initialize the list of successors of the given state 
    succ_states = []

    # loop through each title and its index in the state
    for index, tile in enumerate(state):
        # skip the tile that is not 0
        if tile != 0:
            continue

        # calculate the column and row index of the tile of 0 in the given state
        col, row = index % 3, index // 3

        # when the tile of 0 is not in the fist column 
        if col != 0:
            # copy the state
            state_copy = cp.deepcopy(state)
            # calculate the left index of the 0 tile
            j = 3 * row + (col -1)
            # switch 0 tile with its left tile 
            if state_copy[index] != state_copy[j]:
                temp = state_copy[index]
                state_copy[index] = state_copy[j]
                state_copy[j] = temp
                succ_states.append(state_copy)

        # when the tile of 0 is not in the last column 
        if col != 2:
            # copy the state
            state_copy = cp.deepcopy(state)
            # calculate the right tile index of the 0 tile
            j = 3 * row + (col + 1)
            # switch 0 tile with its right tile 
            if state_copy[index] != state_copy[j]:
                temp = state_copy[index]
                state_copy[index] = state_copy[j]
                state_copy[j] = temp
                succ_states.append(state_copy)

        # when the tile of 0 is not in the first column
        if row != 0:
            # copy the state
            state_copy = cp.deepcopy(state)
            # calculate the above tile index of the 0 tile
            j = 3 * (row - 1) + col
            # switch 0 tile with its above tile 
            if state_copy[index] != state_copy[j]:
                temp = state_copy[index]
                state_copy[index] = state_copy[j]
                state_copy[j] = temp
                succ_states.append(state_copy)

        # when the tile of 0 is not in the first column
        if row != 2:
            # copy the state
            state_copy = cp.deepcopy(state)
            # calculate the under tile index of the 0 tile
            j = 3 * (row + 1) + col
            # switch 0 tile with its under tile
            if state_copy[index] != state_copy[j]:
                temp = state_copy[index]
                state_copy[index] = state_copy[j]
                state_copy[j] = temp
                succ_states.append(state_copy)
   
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    # initialize the max length
    max_length = 0
    # initialize the dictionary that contains key as node state and id as value
    # for tracking the state of the visited nodes
    visited_states = {}
    # initialize the open list that contains the nodes 
    opened = []
    # initialize the closed list that contains the states of the nodes
    closed = {}
    # initalize the dictionary that contains key as id and value as node
    visited = {}
    
    # initial g value
    init_g = 0
    # inital h value
    init_h = get_manhattan_distance(state)
    # intial parent id
    init_parent = -1
    # initial id value
    init_id = 0

    # initialize the initial node with cost(g+h), state, g value, h value, parent id, its node id
    init_node = (init_g + init_h, state, (init_g, init_h, init_parent, init_id))
    # store first node in the open list
    heapq.heappush(opened, init_node)
    # store key as the node state and node as a value in the visited_states dictionary
    # to track the state of the node 
    visited_states[tuple(state)] = init_id
    # store key as the node id and node as a value in the visited list
    visited[init_id] = init_node
    
    # loop while open is not empty
    while True:
        if max_length < len(opened):
            # the max queue length
            max_length = len(opened)
        
        # if open is empty, break
        if not opened:
            break

        # pop the node with lowest cost from opened list
        curr_node = heapq.heappop(opened)
        # move the node with lowest cost to closed list
        (curr_cost, curr_state, (curr_g, curr_h, curr_parent, curr_id)) = curr_node

        # add the state of the node with the lowest cost in the closed dictionary that is poped from opened list
        closed[tuple(curr_state)] = curr_id

        # if the state equals goal state, break
        if curr_state == goal_state:
            break

        # get successor states of the state    
        succ_states = get_succ(curr_state)
        # loop successor states
        for succ_state in succ_states:
            # increment 1 from current g 
            next_g = curr_g + 1

            # when the state is already in the closed list
            if tuple(succ_state) in closed:
                # return id of the state that is already in the closed list
                exist_id = closed[tuple(succ_state)]
                # calculate the h distance of the state
                next_h = get_manhattan_distance(succ_state)
                # return node with the state that is already in the closed list
                exist_node = visited[exist_id]
                # when the cost of successor state that is not yet added to the closed list is lesser than 
                # the node with the same state that has been added to the closed list previously
                if exist_node[0] > next_g + next_h: 
                    # replace the node with the state with lesser cost in the visited list
                    visited[exist_id] = (next_g + next_h, succ_state, (next_g, next_h, curr_id, exist_id))
                    new_node = visited[exist_id]
                    # add the state to the visited states dictionary
                    visited_states[tuple(succ_state)] = exist_id   
                    # remove the state from the closed list
                    del closed[tuple[succ_state]]
                    # push the replaced node to the opened list
                    heapq.heappush(opened, new_node)           

            # when the state is in the opened list
            elif (tuple(succ_state) in visited_states) and (tuple(succ_state) not in closed):
                 # return the id of the state that is already in the opened list
                 exist_id = visited_states[tuple(succ_state)]
                 # calculate the h distance of the state
                 next_h = get_manhattan_distance(succ_state)
                 # return node with the state that is already in the opened list 
                 exist_node = visited[exist_id]
                 # when the cost of successor state that is not yet added to the opened list is lesser than
                 # the node with the same state that has been added to the opened list previously
                 if exist_node[0] > next_g + next_h:
                     # replace the node with the state with lesser cost in the visited list
                     visited[exist_id] = next_g + next_h, succ_state, (next_g, next_h, curr_id, exist_id)
                     new_node = visited[exist_id] 
                     # add the state to the visited states dictionary
                     visited_states[tuple(succ_state)] = exist_id
                     # push the replaced node to the opened list
                     heapq.heappush(opened, new_node)
            # if the node is not in open and closed list, and it is a new node
            else:
                # calculate the h distance of the successor state
                new_h = get_manhattan_distance(succ_state)
                # calculate the id of successor state
                new_id = len(visited) 
                # initialize the the node with its cost, state, g, h, parent id, and its id
                new_node = (next_g + new_h, succ_state, (next_g, new_h, curr_id, new_id))
                # add the new node's id as the key and node as the value to the visited dictionary
                visited[new_id] = new_node
                # add the state of this new node to the visited states dictionary
                visited_states[tuple(succ_state)] = new_id
                # push the new node to the opend list
                heapq.heappush(opened, new_node)

    # add the node's current state, current h and current move that is initially added to the closed list
    state_info_list = [(curr_state, curr_h, curr_g)]
    # when the state meets the goal in the while loop and it breaks
    if curr_state == goal_state:
        # loop until it reaches the initial node with initial state
        while curr_parent != -1:
            # the parent node of the current node
            node_tracker = visited[curr_parent]
            curr_cost, curr_state, (curr_g, curr_h, curr_parent, curr_id) = node_tracker
            # add the parent node in the state_info_list
            state_info_list.append((curr_state, curr_h, curr_g))
        # reverse its sequence so it prints from the initial node with initial state to the node with goal state
        state_info_list = reversed(state_info_list)
    # when the state doesn't meet the goal state, return
    else:
        return

    # This is a format helper.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    for state_info in state_info_list:
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print()

    print(get_manhattan_distance([6, 0, 0, 3, 5, 1, 7, 2, 4], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print()
