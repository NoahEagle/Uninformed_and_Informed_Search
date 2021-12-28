############################################################
# CIS 521: Informed Search Homework
############################################################

student_name = "Noah Eagle"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

import math

import queue

import copy

import random

############################################################
# Section 1: Tile Puzzle
############################################################

found_solution = False

# Returns a new TilePuzzle of the specified dimensions in the starting
# configuration (tiles arranged in ascending order from left to right, top to
# bottom except for the bottom right coord which is 0 for empty tile)
def create_tile_puzzle(rows, cols):

    # Create an initially empty board for our tile puzzle
    tile_puzzle = []

    # Add in the appropriate number of rows to our board list
    for i in range (rows):
        tile_puzzle.append([])

    # Initialize a counter so we know which number to assign to the next tile
    next_number = 1

    # Iterate through the board, assigning numbers to the tiles in ascending
    # order (top to bottom, left to right)
    for i in range (rows):
        for j in range (cols):
            tile_puzzle[i].append(next_number)
            next_number += 1

    # Reset that bottom right coord to 0 for the empty tile
    tile_puzzle[rows - 1][cols - 1] = 0

    # Now create a TilePuzzle instance with this board and return it
    return TilePuzzle(tile_puzzle)

class TilePuzzle(object):
    
    # Initializes an instance of the TilePuzzle class with a given board input
    # (it also stores the row and col count of the board and the location of the
    # empty tile)
    def __init__(self, board):

        # Store the tile puzzle board
        self.gameboard = board

        # Store the row and col count of the board
        self.board_rows = len(board)
        self.board_cols = len(board[0])

        # Iterate through the board to find the coordinate of the empty tile (0)
        for i in range (len(board)):
            for j in range (len(board[i])):
                if board[i][j] == 0:
                    self.empty_tile_coord = (i, j)

        self.found_solutions = False

    # Returns the internal representation of the tile puzzle board
    def get_board(self):
        return self.gameboard

    # Attempts to swap the empty tile with an adjacent neighbor (up, down, 
    # left, right). If the direction input is invalid or we can't make the
    # desired move, we return False. If we successfully execute a move, we
    # return True
    def perform_move(self, direction):

        # Fetch the row and col components of the empty tile coord
        empty_row, empty_col = self.empty_tile_coord

        gameboard = self.gameboard

        # If the desired move direction is up...
        if direction == "up":

            # And we can actually move up...
            if empty_row > 0:

                # Set the current empty tile spot to be the tile just above it
                gameboard[empty_row][empty_col] = \
                gameboard[empty_row - 1][empty_col]

                # Then set the tile above the old empty tile spot to be empty
                gameboard[empty_row - 1][empty_col] = 0

                # Update the board's empty tile field to the one above
                self.empty_tile_coord = (empty_row - 1, empty_col)

                # Return True as we successfully executed the move
                return True

            # If we can't move up...
            else:

                # Return False as the move is impossible
                return False

        # If the desired move direction is down...
        elif direction == "down":

            # And we can actually move down...
            if empty_row < self.board_rows - 1:

                # Set the current empty tile spot to be the tile spot just below
                gameboard[empty_row][empty_col] = \
                gameboard[empty_row + 1][empty_col]

                # Then set the tile below the old empty tile spot to be empty
                gameboard[empty_row + 1][empty_col] = 0

                # Update the board's empty tile field to the one below
                self.empty_tile_coord = (empty_row + 1, empty_col)

                # Return True as we successfully executed the move
                return True

            # If we can't move down...
            else:

                # Return False as the move is impossible
                return False

        # If the desired move direction is left...
        elif direction == "left":

            # And we can actually move left...
            if empty_col > 0:

                # Set the current empty tile spot to be the value to the left
                gameboard[empty_row][empty_col] = \
                gameboard[empty_row][empty_col - 1]

                # Then set the left tile to be empty
                gameboard[empty_row][empty_col - 1] = 0

                # Update the board's empty tile field to the one just left
                self.empty_tile_coord = (empty_row, empty_col - 1)

                # Return True as we successfully executed the move
                return True

            # If we can't move left...
            else:

                # Return False as the move is impossible
                return False

        # If the desired move direction is right
        elif direction == "right":

            # And we can actually move right
            if empty_col < self.board_cols - 1:

                # Set the current empty tile spot to be the value to the right
                gameboard[empty_row][empty_col] = \
                gameboard[empty_row][empty_col + 1]

                # Then set the right tile to be empty
                gameboard[empty_row][empty_col + 1] = 0

                # Update the board's empty tile field to the one just right
                self.empty_tile_coord = (empty_row, empty_col + 1)

                # Return True as we successfully executed the move
                return True

            # If we can't move right...
            else:

                # Return False as the move is impossible
                return False

        # Assuming the direction wasn't "up", "down", "left", or "right", just
        # return False as we won't be able to make any move then
        return False

    # Scrambles the puzzle by calling perform_move(self, direction) the
    # indicated number of times (guaranteeing that the resulting configuration
    # will be solvable)
    def scramble(self, num_moves):

        # For each one of the desired number of moves
        for i in range(num_moves):

            # Choose a random direction
            random_direction = random.choice(["up", "down", "left", "right"])

            # Perform a move in that direction
            TilePuzzle.perform_move(self, random_direction)

    # Returns whether the board is in its starting (goal) configuration
    # (ascending numerical order starting at 1 from top to bottom, left to right
    # with a 0 in the bottom right corner)
    def is_solved(self):

        # Retrieve the internal representation of the game board
        gameboard = self.gameboard

        # Initialize an int to serve as the previous number that the current
        # tile should be one greater than
        prev_number = 0

        # Iterate through the board
        for i in range(self.board_rows):
            for j in range(self.board_cols):

                # If the current coord is not the bottom right
                if not (i == self.board_rows - 1 and j == self.board_cols - 1):
                    
                    # And if the current tile is not one greater than the
                    # previous tile, return False (wrong ordering)
                    if (gameboard[i][j] != prev_number + 1):
                        return False

                # If we are in the bottom right
                else:

                    # If the tile is not empty, return False (wrong ordering)
                    if gameboard[i][j] != 0:
                        return False

                prev_number += 1

        # If we made it through the board and detected no anomalies, we're safe
        # to return True (we're in a correct ordering)
        return True

    # Returns a new TilePuzzle object initialized with a deep copy of the
    # current board
    def copy(self):

        # Grab the gameboard for this current TilePuzzle object
        gameboard = self.gameboard

        # Make a deep copy of this gameboard
        deepcopied_gameboard = copy.deepcopy(gameboard)

        # Create and return a new TilePuzzle object with this deep copied board
        return TilePuzzle(deepcopied_gameboard)

    # Yields all successors of the puzzle as (direction, new-puzzle) tuples
    # (where the new-puzzle component is a TilePuzzle object whose board is
    # the result of applying the corresponding move to the current board)
    def successors(self):

        # Create four deep copies of the current puzzle (one for each direction)
        deepcopied_puzzle1 = self.copy()
        deepcopied_puzzle2 = self.copy()
        deepcopied_puzzle3 = self.copy()
        deepcopied_puzzle4 = self.copy()

        # If we can move upwards, do so and yield the corresponding tuple
        if (deepcopied_puzzle1.perform_move("up")):
            yield ("up", deepcopied_puzzle1)

        # If we can move downwards, do so and yield the corresponding tuple
        if (deepcopied_puzzle2.perform_move("down")):
            yield ("down", deepcopied_puzzle2)

        # If we can move leftwards, do so and yield the corresponding tuple
        if (deepcopied_puzzle3.perform_move("left")):
            yield ("left", deepcopied_puzzle3)

        # If we can move rightwards, do so and yield the corresponding tuple
        if (deepcopied_puzzle4.perform_move("right")):
            yield ("right", deepcopied_puzzle4)

    # A helper function that performs a depth limited search (so dfs but only
    # up to a maximum specified depth)
    def iddfs_helper(self, limit, moves):

        # If we've reached/surpassed the depth limit, return (we shouldn't be
        # searching further)
        if limit <= 0:
            return

        # Decrement the remaining depth limit by 1 as we're now going to explore
        # a node's children
        limit -= 1

        # For each (move, successor) of the current puzzle
        for move, child_puzzle in self.successors():

            # Create the move list to the child (by just adding the last move
            # to the current move list)
            new_move_list = moves + [move]

            # If the child is a goal state, yield its move list as a
            # solution
            if child_puzzle.is_solved():
                yield new_move_list

            # If the child isn't a goal state, we have more work to do
            else:

                # Continue the depth limit search from the child
                yield from child_puzzle.iddfs_helper(limit, new_move_list)

    # Yields all optimal solutions to the board using Iterative Deepening DFS
    def find_solutions_iddfs(self):

        # If the puzzle is already solved, yield the empty list as our solution
        # (we need no moves for an optimal solution)
        if self.is_solved():
            yield []
            return

        # Initialize the depth limit to 1 (as we just handled the 0 depth case)
        depth_limit = 1

        # Initialize a flag for whether we've found the optimal solution depth
        # as False
        solved = False

        # The iterative deepening dfs process
        while True:

            # Perform a depth limited search at the current depth limit
            # (move list is empty as we're starting from the initial state)
            results = self.iddfs_helper(depth_limit, [])

            # Do the depth limited search again so we can use one result to
            # check if we found a solution and the other to yield the solutions
            # (THIS IS A SCUFFED WORKAROUND TO KNOW WHEN TO BREAK OUT OF LOOP)
            results_copy = self.iddfs_helper(depth_limit, [])

            # If we would be yielding at least one solution at this depth,
            # we've found the shallowest depth solution(s), so flag solved as
            # True
            if len(list(results_copy)) != 0:
                solved = True

            # Now yield the results of our search (either a simple return for
            # no solutions or a bunch of move lists if we found the optimal
            # solutions)
            yield from results

            # If we had found solutions at that depth, flip the solved tag to
            # True and break out of this while loop (to stop the iterative 
            # deepening DFS process)
            if solved:
                break

            # If we didn't find any solutions, increment the depth limit by 1
            # so our next DFS will search one depth further
            depth_limit += 1

    # A helper function to estimate the distance between a specific board state
    # and the goal state (distance being estimated via the sum of the
    # Manhattan distances between each tile's actual position and goal position)
    def heuristic_estimate(self):

        # Grab the current board
        gameboard = self.gameboard

        # Create a dictionary to store mappings between tiles and their goal
        # positions
        correct_placements = {}

        # Initialize an int to serve as the number that should be appearing next
        next_num = 1

        # Iterate through the board making mappings such that the next ascending
        # tile number should be placed in the next coord assuming we're going
        # left to right, top to bottom
        for i in range(self.board_rows):
            for j in range(self.board_cols):
                correct_placements[next_num] = (i, j)
                next_num += 1

        # Create a dictionary to store mappings between tiles and their current
        # positions
        actual_placements = {}

        # Iterate through the board to find each tile's current position and
        # create the relevant mapping
        for i in range(self.board_rows):
            for j in range(self.board_cols):
                actual_placements[gameboard[i][j]] = (i, j)

        # Initialize the manhattan distance estimate to 0
        manhattan_dist = 0

        # For each of the numbered tiles (so just not 0, the empty tile)
        for tile_num in range (1, (self.board_rows * self.board_cols) - 1):

            # Find the goal coordinates
            correct_row, correct_col = correct_placements[tile_num]

            # Find the current coordinates
            actual_row, actual_col = actual_placements[tile_num]

            # Add this tile's distance to the running Manhattan distance total
            manhattan_dist += \
            abs(correct_row - actual_row) + abs(correct_col - actual_col)

        # After we get through all the tiles, we're clear to return the total
        # Manhattan distance heuristic estimate
        return manhattan_dist

    # Returns an optimal move list to get from the current board to the goal
    # configuration using A* search with a Manhattan distance heuristic.
    def find_solution_a_star(self):

        # Create a priority queue to serve as our frontier
        Prio_Queue = queue.PriorityQueue()

        # Insert this initial state into the priority queue
        # (prio, path cost, move_list, puzzle)
        Prio_Queue.put((0, 0, [], self))

        # Create a set to keep track of the states we've visited
        explored_set = set()

        # Add this initial state to the explored set (as a tuple of tuples for
        # better lookup)
        explored_set.add(tuple(tuple(row) for row in self.gameboard))

        # So long as the priority queue is not empty
        while not Prio_Queue.empty():

            # Fetch the best state from the priority queue
            curr_prio, curr_path_cost, curr_move_list, curr_puzzle = \
            Prio_Queue.get()

            # If we're in a goal state, return the move list as our solution
            if TilePuzzle.is_solved(curr_puzzle):
                return curr_move_list

            # If we're not in a goal state, we've more work to do
            else:

                # For each successor of this current puzzle
                for move, child_puzzle in TilePuzzle.successors(curr_puzzle):

                    # Assuming we haven't seen this state already
                    if (tuple(tuple(row) for row in child_puzzle.gameboard)) not in explored_set:

                        # Add this state to the explored set
                        explored_set.add(tuple(tuple(row) for row in child_puzzle.gameboard))

                        # Calculate the path cost to this state by adding one 
                        # (for the move from current to child)
                        new_path_cost = curr_path_cost + 1

                        # Calculate the estimated distance to the goal state
                        heuristic_estimate = TilePuzzle.heuristic_estimate(child_puzzle)

                        # Calculate the priority by adding g(n) and h(n)
                        prio = new_path_cost + heuristic_estimate

                        # Create a copy of the current move list
                        new_move_list = copy.deepcopy(curr_move_list)

                        # Add this latest move to the copy of the move list
                        new_move_list.append(move)

                        # Insert this child state into the priority queue with
                        # the proper priority, path cost, move list, and puzzle
                        Prio_Queue.put((prio, new_path_cost, new_move_list, child_puzzle))

        # If we exhausted the priority queue and didn't find a solution, then
        # one doesn't exist, so return None
        return None

############################################################
# Section 2: Grid Navigation
############################################################

# Returns the estimated cost from this current coord to the goal coord
def find_heuristic_value(curr_coord, goal_coord):

    # Grab the current row and col
    curr_row, curr_col = curr_coord

    # Grab the goal row and col
    goal_row, goal_col = goal_coord

    # Calculate the straight line distance from the current coord to goal coord
    # (classic Pythagorean Theorem) and return it
    return math.sqrt((abs(curr_row - goal_row))**2 + (abs(curr_col - goal_col))**2)

def find_successors(curr_coord, scene):

    # Grab the current row and col
    curr_row, curr_col = curr_coord

    # If we can move up a row in the board (w/o going off)
    if curr_row > 0:

        # If directly above is False (meaning empty)
        if not scene[curr_row - 1][curr_col]:

            # Yield that coord as a successor state
            yield (curr_row - 1, curr_col)

        # If we can also move left a col in the board (w/o going off)
        if curr_col > 0:

            # If directly up and left is False (meaning empty)
            if not scene[curr_row - 1][curr_col - 1]:

                # Yield that coord as a successor state
                yield (curr_row - 1, curr_col - 1)
    
        # If we can also move right a col in the board (w/o going off)
        if curr_col < len(scene[0]) - 1:

            # If directly up and right is False (meaning empty)
            if not scene[curr_row - 1][curr_col + 1]:

                # Yield that coord as a successor state
                yield(curr_row - 1, curr_col + 1)

    # If we can move down a row in the board (w/o going off)
    if curr_row < len(scene) - 1:

        # If directly below is False (meaning empty)
        if not scene[curr_row + 1][curr_col]:

            # Yield that coord as a successor state
            yield (curr_row + 1, curr_col)

        # If we can also move left a col in the board (w/o going off)
        if curr_col > 0:

            # If directly down and left is False (meaning empty)
            if not scene[curr_row + 1][curr_col - 1]:

                # Yield that coord as a successor state
                yield (curr_row + 1, curr_col - 1)

        # If we can also move right a col in the board (w/o going off)
        if curr_col < len(scene[0]) - 1:

            # If directly down and right is False (meaning empty)
            if not scene[curr_row + 1][curr_col + 1]:

                # Yield that coord as a successor state
                yield (curr_row + 1, curr_col + 1)

    # If we can move left a col in the board (w/o going off)
    if curr_col > 0:

        # If directly to the left is False (meaning empty)
        if not scene[curr_row][curr_col - 1]:

            # Yield that coord as a successor state
            yield (curr_row, curr_col - 1)

    # If we can move right a col in the board (w/o going off)
    if curr_col < len(scene[0]) - 1:

        # If directly to the right is False (meaning empty)
        if not scene[curr_row][curr_col + 1]:

            # Yield that coord as a successor state
            yield (curr_row, curr_col + 1)

# Returns the shortest path from the start coord to the goal coord in a 2D grid
# (scene) in which some coords contains obstacles (meaning those coords
# cannot be traversed). Obstacle coords are "True" and free/empty coords are
# "False". Allowed movements are up, up-left, up-right, right, down-right, down,
# down-left, left. The shortest path is calculated using A*
def find_path(start, goal, scene):

    # Grab the start row and col
    start_row, start_col = start

    # If the start coord is True (meaning is on an obstacle), return None
    # (there's no solution possible)
    if scene[start_row][start_col]:
        return None

    # Grab the goal row and col
    goal_row, goal_col = goal

    # If the goal coord is True (meaning is on an obstacle), return None
    # (there's no solution possible)
    if scene[goal_row][goal_col]:
        return None

    # If we're already at the goal coord, just return an empty list (as we don't
    # need any moves for our solution)
    if start == goal:
        return []

    # Create an initially empty priority queue for our frontier
    Prio_Queue = queue.PriorityQueue()

    # Put the start state and its path cost (0) into the Prio Queue w/ prio 1
    # (although prio here doesn't matter as it'll get taken off first no
    # matter what b/c it's the only thing on when we start)
    Prio_Queue.put((1, 0, start))

    # Create an initially empty dict to keep track of the current shortest 
    # distances to each discovered coord
    cost_so_far = {}

    # Add this start coord to the current shortest path cost dict with a value
    # of 0 (as we start there)
    cost_so_far[start] = 0

    # Create an initially empty dictionary to store mappings between child
    # states (coords) and parent states (the coords we traveled from to get
    # to child). Mappings will be of form 
    # (child_row, child_col): (parent_row, parent_col)
    child_to_parent_dict = {}

    # Add a mapping for the initial state with a parent of None (as start
    # state doesn't have a parent)
    child_to_parent_dict[start] = None

    # So long as the Prio Queue isn't empty
    while not Prio_Queue.empty():

        # Grab the highest prio state off the Prio Queue
        curr_state_info = Prio_Queue.get()

        # Extract the curr coord and curr path cost to that coord
        curr_prio, curr_path_cost, curr_coord = curr_state_info

        # If this coord is the goal, then we've made it
        if curr_coord == goal:

            # Create an initially empty list to serve as our move list for
            # this found solution
            solution_move_list = []

            # Add this final move to the solution move list
            solution_move_list.append(curr_coord)

            # This child coord's parent is the current coord we were dealing w/
            parent_coord = curr_coord

            # So long as we haven't reached the initial coord (state state)
            while child_to_parent_dict[parent_coord] != None:

                # Find the next parent coord moving backwards in our solution
                new_parent_coord = child_to_parent_dict[parent_coord]

                # Add this prev parent coord to the move solution list
                solution_move_list.append(new_parent_coord)

                # Set the parent coord to this new parent coord
                parent_coord = new_parent_coord

            # After we've gone back up to the initial coord, we need to reverse
            # the move list because we assembled it backwards
            solution_move_list = solution_move_list[::-1]

            # Now we're clear to return the move list as our solution
            return solution_move_list

        # If this coord isn't the goal, then we still have work to do
        else:

            # Examine each successor state for this current coord (state)
            for successor_coord in find_successors(curr_coord, scene):

                successor_row, successor_col = successor_coord

                curr_row, curr_col = curr_coord

                # Update the path cost to this successor to be the
                # path cost to the current coord + dist for the move to
                # successor
                new_path_cost = curr_path_cost + \
                math.sqrt((abs(successor_row - curr_row))**2 + 
                (abs(successor_col - curr_col))**2)

                # If we haven't already explored this coord or its path cost is 
                # lower than the shortest path cost we've found so far to that 
                # coord
                if successor_coord not in cost_so_far or \
                new_path_cost < cost_so_far[successor_coord]:

                    # Add/update its shortest path cost from start
                    cost_so_far[successor_coord] = new_path_cost

                    # Add a child-parent mapping for this successor to its
                    # parent (the current coord we were working with)
                    child_to_parent_dict[successor_coord] = curr_coord

                    # Calculate h(n) for n = successor_coord
                    heuristic_value = \
                    find_heuristic_value(successor_coord, goal)

                    # Calculate f(n), priority, by adding h(n) and g(n)
                    prio = heuristic_value + new_path_cost

                    # Place this successor state into the Prio Queue
                    # w/ the proper prio value
                    Prio_Queue.put((prio, new_path_cost, successor_coord))
    
    # If we exhaust the Prio Queue and never found the goal node, then there's
    # no available solution, so return None
    return None

############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

# A helper function to test whether a board is solved for the distinct disk
# problem
def dist_is_sovled(n, board):

    # For final n spots (excluding the very final spot) of the board
    for i in range(n):

        # If any of the spots are not 1 number higher than their right neighbor
        if not (board[len(board) - 1 - i] == i):

            # This is not a goal state (original disks are not now in reverse
            # order on the right side), so return False
            return False

    # If each spot satisfied the condition, then we are in reverse order on
    # the right side, so return True (this is a goal state)
    return True

# A helper function that yields the successors of the current board (state) in
# the form (move, resulting state) where the move is a tuple of the form 
# (start, end) for the moved disk and the resulting state is another board
def dist_successors(board):

    # For each spot in the board (except the final spot)
    for i in range(len(board) - 1):

        # If the current spot is filled (not -1) and the next spot is empty
        if board[i] != -1 and board[i + 1] == -1:

            # Make a deepcopy of the board
            deepcopied_board = copy.deepcopy(board)

            # Place the distinct disk in the next spot
            deepcopied_board[i + 1] = board[i]

            # Make the old spot empty
            deepcopied_board[i] = -1

            # Yield this move and its resulting board (state)
            yield ((i, i + 1), deepcopied_board)

    # For each spot in the board (except the final two spots)
    for i in range(len(board) - 2):

        # If the current spot is filled and the next spot is also filled and
        # two spots forward is empty
        if board[i] != -1 and board[i + 1] != -1 and board[i + 2] == -1:

            # Make a deepcopy of the board
            deepcopied_board = copy.deepcopy(board)

            # Place the distinct disk two spots down
            deepcopied_board[i + 2] = deepcopied_board[i]

            # Make the old spot empty
            deepcopied_board[i] = -1

            # Yield this move and its resulting board (state)
            yield ((i, i + 2), deepcopied_board)

    # For all spots in the board (except the first)
    for i in range(1, len(board)):

        # If the current spot is filled and the previous spot is empty
        if board[i] != -1 and board[i - 1] == -1:

            # Make a deepcopy of the board
            deepcopied_board = copy.deepcopy(board)

            # Place the distinct disk in the prev spot
            deepcopied_board[i - 1] = board[i]

            # Make the old spot empty
            deepcopied_board[i] = -1

            # Yield this move and its resulting board (state)
            yield ((i, i - 1), deepcopied_board)

    # For all spots in the board (except the first two)
    for i in range(2, len(board)):

        # If the current spot is filled, the previous spot is filled, but
        # two spaces previously isn't filled
        if board[i] != -1 and board[i - 1] != -1 and board[i - 2] == -1:

            # Make a deepcopy of the board
            deepcopied_board = copy.deepcopy(board)

            # Place the distinct disk in the next spot
            deepcopied_board[i - 2] = board[i]

            # Make the old spot empty
            deepcopied_board[i] = -1

            # Yield this move and its resulting board (state)
            yield ((i, i - 2), deepcopied_board)

# A helper function to find the estimated cost of getting from a certain state
# to the goal state (w/ the estimation being the sum of the number of spots
# each distinct disk is out of place by divided by two as we're loosening
# the game restrictions to allow for each disk to move two spots at a time
# and can move onto occupied spots)
def find_heuristic_estimate(n, board):

    # Initialize the estimate to 0
    h = 0

    # Iterate through the board
    for x in range(0, len(board) - 1):

        # If the current spot has a disk (as empty spots are -1)
        if board[x] != -1:

            # Add the distance by which that disk is out of place to the
            # running total estimate
            h = h + abs(len(board) - 1 - board[x] - x)

    # Now return our out-of-place estimate divided by 2 (b/c we're estimating
    # out of place assuming each disk can always move 2 spots away at a time)
    return h / 2

# Returns an optimal solution to the problem in which there are numbered
# (distinct) disks in the first n spots of a 1D grid and we need to move them
# into the final n spots in reverse order (so if it was 1, 2, 3 before, it now
# needs to end as 3, 2, 1). Solutions are returned as move lists in which the
# moves have the form (start, end) detailing from where a disk was taken and 
# where it was placed. We're searching via A* here.
def solve_distinct_disks(length, n):

    # Create a list (initially empty) to serve as the starting configuration
    initial_board = []

    # Fill the first n spots with distinct disks (labeled 0 to n - 1)
    for i in range(n):
        initial_board.append(i)

    # Leave the remaining spots empty (filled with a -1)
    for i in range(n, length):
        initial_board.append(-1)

    # If this initial configuration is a goal state, return the empty list
    # (as we need no moves to solve the game)
    if dist_is_sovled(n, initial_board):
        return []

    # Create an initially empty Prio Queue
    Prio_Queue = queue.PriorityQueue()

    # Place this initial configuration into the Prio Queue 
    # (prio 0, state, curr path cost)
    Prio_Queue.put((0, initial_board, 0))

    # Create an initially empty set (to track which states we've seen)
    explored_set = set()

    # Add this initial configuration to the explored set (as a tuple for easier
    # set membership checks)
    explored_set.add(tuple(initial_board))

    # Create an initially empty child-to-parent mapping which will contain
    # mappings of the form child-state: (move from parent, parent-state)
    # where the states are boards (tuples) and the moves are tuples of the form
    # (start, end) for a single disk movement
    child_to_parent_dict = {}

    # Add a mapping for the initial_board from its parent (with its 
    # corresponding move). Since there is no parent/move, just put (None, None)
    child_to_parent_dict[tuple(initial_board)] = (None, None)

    # So long as the Prio Queue has not been exhausted
    while not Prio_Queue.empty():

        # Grab the next state to explore
        curr_board_info = Prio_Queue.get()

        # Grab the curr board for this state
        curr_board = curr_board_info[1]

        # If this board is a goal state, we've made it
        if dist_is_sovled(n, curr_board):

            # Create an initially empty list to store the move list for our
            # found solution
            solution_move_list = []

            # Find out what move and parent state we came from
            prev_move, parent_board = child_to_parent_dict[tuple(curr_board)]

            # Add this final move to the solution move list
            solution_move_list.append(prev_move)

            # So long as we haven't hit the initial state as the next parent
            while child_to_parent_dict[tuple(parent_board)] != (None, None):

                # Grab the parent and corresponding move that took us to
                # this state
                prev_move, new_parent = \
                child_to_parent_dict[tuple(parent_board)]

                # Add the move to the solutions move list
                solution_move_list.append(prev_move)

                # Update the parent to this new one
                parent_board = new_parent

            # After we've added all moves from the start state, we need to
            # reverse the list as we assembled it backwards (from goal
            # state to start state)
            solution_move_list = solution_move_list[::-1]

            # Now we're clear to return this move list as our solution
            return solution_move_list

        # For each successor of the current board (state)
        for (move, new_board) in dist_successors(curr_board):

            # Assuming this board (state) wasn't encountered before
            if tuple(new_board) not in explored_set:

                # Calculate the estimated distance from successor to goal
                h = find_heuristic_estimate(n, new_board)

                # Grab the current path cost to this state
                curr_path_cost = curr_board_info[2]

                # For this successor, give it a path cost of current cost + 1
                # (since all moves cost 1)
                new_path_cost = curr_path_cost + 1

                # Calculate f(n), prio, by summing the heuristic value and 
                # the path cost
                prio = h + new_path_cost

                # Put it in the Prio Queue
                Prio_Queue.put((prio, new_board, new_path_cost))

                # Add it to the explored set
                explored_set.add(tuple(new_board))

                # Set its child: (move from parent, parent) mapping
                child_to_parent_dict[tuple(new_board)] = \
                (move, tuple(curr_board))

    # If we exhausted the Prio Queue and never found a solution, this game is
    # unsolvable (so return None)
    return None