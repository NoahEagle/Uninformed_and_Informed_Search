# Uninformed_and_Informed_Search
Solves two puzzles/games using the A* informed search algorithm and another using the Iterative Deepening Depth-First Search uninformed algorithm.

The first puzzle is the tile puzzle (solved with IDDFS). You have a grid layout (3x3 for example) with one empty spot. 
The other spots are filled with consecutive numbers ascending from 1 (so 1 through 8 in the 3x3 game). 
Initially, the tile placements are randommized. 
The goal is to rearrage the tiles (by shifting one tile adjacent to the empty into the empty spot at a time) so that the tiles are arranged in ascending order if you were reading from top to bottom, left to right with the final bottom right corner being the empty spot. For the 3x3 game, you'd want 1 2 3 in the top row, 4 5 6 in the middle row, and 7 8 empty in the bottom row.

The final two puzzles (solved with A*) are grid navigation (in which you want to find the shortest path between two points in a grid with obstacles and in which you can move up, down, left, right, and in any of the diagonals) and linear distinct disks (in which you have a row of l cells with the first n being occupied by distinct disks, and in which your goal is to move these distinct disks via moving one forward at a time into an empty space or having one jump at a time over one disk just in front to an empty spot two cells ahead until the disks are arranged in reverse order in the final n cells of the row).
