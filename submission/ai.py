import time
import math
import random
import numpy as np
from typing import List, Tuple, Set
from helper import *

def check_non_adjacent_neighbours(connected_neighbours: List[Tuple[int, int]], neighbours: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
    for i in range(len(connected_neighbours)):
        for j in range(i + 1, len(connected_neighbours)):
            if connected_neighbours[j] not in neighbours[connected_neighbours[i]]:
                return True
    return False

def precompute_board_data(dim: int) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    neighbours = {}
    corners = set(get_all_corners(dim))
    edges = set()
    for edge_list in get_all_edges(dim):
        edges.update(edge_list)
    
    for i in range(dim):
        for j in range(dim):
            neighbours[(i, j)] = get_neighbours(dim, (i, j))
    
    return neighbours, corners, edges

# Optimized check_fast_win function
def check_fast_win(board: np.array, move: Tuple[int, int], player_num: int, neighbours: Dict[Tuple[int, int], List[Tuple[int, int]]], corners: Set[Tuple[int, int]], edges: Set[Tuple[int, int]]) -> bool:
    board_bool = (board == player_num)
    connected_neighbours = [n for n in neighbours[move] if board_bool[n]]
    l=len(connected_neighbours)
    b=False
    if l < 1:
        return False
    
    if move in corners:
        if check_bridge(board_bool, move):
            return True
    
    if move in edges:
        if check_fork(board_bool, move):
            return True
    if (l>=2)  and check_non_adjacent_neighbours(connected_neighbours, neighbours):
        b = check_ring(board_bool, move)
        if b:
            return True
        # print("fork and bridge and ring")
        win,_ = check_fork_and_bridge(board_bool, move) 
        if win:
            return True
    return False

def can_win_in_one_move(state: np.array, current_player: int,neighbours,corners,edges) -> Tuple[bool, Tuple[int, int]]:
    dim = state.shape[0]
    # neighbours, corners, edges = precompute_board_data(dim)

    for i in range(dim):
        for j in range(dim):
            if state[i, j] == 0:  # Empty cell
                # Make a move
                new_state = state.copy()
                new_state[i, j] = current_player
                
                # Check if this move wins the game
                if check_fast_win(new_state, (i, j), current_player, neighbours, corners, edges):
                    return True, (i, j)
    
    return False, None

def mate_in_two(state: np.array, current_player: int,neighbours,corners,edges) -> Tuple[bool, Tuple[int, int]]:
    dim = state.shape[0]
    # neighbours, corners, edges = precompute_board_data(dim)

    for i in range(dim):
        for j in range(dim):
            if state[i, j] == 0:  # Empty cell
                # Make a move
                new_state = state.copy()
                new_state[i, j] = current_player
                
                # Check if this move creates at least two winning opportunities
                winning_moves = []
                for ni in range(dim):
                    for nj in range(dim):
                        if new_state[ni, nj] == 0:  # Empty cell in the new state
                            next_state = new_state.copy()
                            next_state[ni, nj] = current_player
                            if check_fast_win(next_state, (ni, nj), current_player, neighbours, corners, edges):
                                winning_moves.append((ni, nj))
                                if len(winning_moves) >= 2:
                                    return True, (i, j)
    
    return False, None


def mate_in_three(state: np.array, current_player: int,neighbours,corners,edges) -> Tuple[bool, Tuple[int, int]]:
    dim = state.shape[0]
    opponent = 3 - current_player  # Assuming players are 1 and 2
    # neighbours, corners, edges = precompute_board_data(dim)

    for i in range(dim):
        for j in range(dim):
            if state[i, j] == 0:  # Empty cell
                # Make the first move
                new_state = state.copy()
                new_state[i, j] = current_player
                
                # Check if this move creates a mate in two situation
                if mate_in_two(new_state, current_player,neighbours,corners,edges)[0]:

                    # If not, check all possible opponent responses
                    opponent_cant_prevent = True
                    for oi in range(dim):
                        for oj in range(dim):
                            if new_state[oi, oj] == 0:  # Empty cell for opponent
                                opponent_state = new_state.copy()
                                opponent_state[oi, oj] = opponent
                                
                                # Check if we still have a mate in two after opponent's move
                                if not mate_in_two(opponent_state, current_player,neighbours,corners,edges)[0]:
                                    opponent_cant_prevent = False
                                    break
                        if not opponent_cant_prevent:
                            break
                    
                    if opponent_cant_prevent:
                        return True, (i, j)
    
    return False, None

def player_move(state: np.array) -> Tuple[int, int]:
    player1 = 0
    player2 = 0
    dim = state.shape[0]
    for i in range(dim):
        for j in range(dim):
            if state[i, j] == 1:
                player1 += 1
            if state[i, j] == 2:
                player2 += 1
    return player1, player2

class Node:
    def __init__(self, state, parent=None, move=None, player=1):
        self.state = state.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.player = player
        self.untried_moves = get_valid_actions(state)

import time
class MCTS:
    def __init__(self, player_number, timer):
        self.player_number = player_number
        self.opponent_number = 3 - player_number
        self.timer = timer
        self.C = 1.41  # UCT constant
        self.dim = 11  # Assuming the board dimension is 11x11
        self.neighbours, self.corners, self.edges = precompute_board_data(self.dim)

    
    def get_move(self, state):
        root = Node(state, player=self.player_number)
        if self.dim != state.shape[0]:
            self.dim = state.shape[0]
            self.neighbours, self.corners, self.edges = precompute_board_data(self.dim)
        time_current = time.time()
        player1, player2 = player_move(state)
        # print("player1 ",player1," player2 ",player2)
               # print(player_move(state))
        if self.dim == 7 or player1+player2>7:
            win, move = can_win_in_one_move(state, self.player_number,self.neighbours, self.corners, self.edges)
            if win:
                return move, [], 0
            win, move = can_win_in_one_move(state, self.opponent_number,self.neighbours, self.corners, self.edges)
            if win:
                return move, [], 0
            print("not m1")
            win, move = mate_in_two(state, self.player_number,self.neighbours, self.corners, self.edges)
            if win:
                return move, [], 0
            win, move = mate_in_two(state, self.opponent_number,self.neighbours, self.corners, self.edges)
            if win:
                return move, [], 0
            win, move = mate_in_three(state, self.player_number,self.neighbours, self.corners, self.edges)
            print("not m2")
            if win:
                return move, [], 0
            win, move = mate_in_three(state, self.opponent_number,self.neighbours, self.corners, self.edges)
            if win:
                return move, [], 0
            print("not m3")
         


        if self.dim == 7:
            if player1 == 0 and self.player_number == 1:
                return (0,3), [], 0
            if player1 == 1 and self.player_number == 1:
                if state[1,3] == 2:
                    return (0,2), [], 0
                for i in range(self.dim):
                    for j in range(self.dim//2):
                        # print(state[i,j])
                        if state[i,j] == 2:
                            return (1,4), [], 0
                return (1,2), [], 0
            if player1 == 2 and self.player_number == 1:
                if state[1,4]==1:
                    if state[0,6]==0:
                        return (0,6), [], 0
                    if state[2,5]==0:
                        return (2,5), [], 0
                if state[1,2]==1:
                    if state[0,0]==0:
                        return (0,0), [], 0
                    if state[2,1]==0:
                        return (2,1), [], 0
            if player1 == 3 and self.player_number == 1:
                if state[2,5]==1:
                    if state[3,6] == 0:
                        return (3,6), [], 0
                    if state[4,4] == 0:
                        return (4,4), [], 0
                if state[2,1]==1:
                    if state[3,0] == 0:
                        return (3,0), [], 0
                    if state[4,2] == 0:
                        return (4,2), [], 0
            
            if player1 == 1 and self.player_number == 2:
                if state[0,0]==1:
                    return (1,1), [], 0
                elif state[0,3]==1:
                    return (1,3), [], 0
                elif state[0,6]==1:
                    return (1,5), [], 0
                elif state[3,6]==1:
                    return (3,5), [], 0
                elif state[6,3]==1:
                    return (5,3), [], 0
                elif state[3,0]==1:
                    return (3,1), [], 0


        if self.dim == 11:
            lst=[
                    [[(5,0)],[[(6,1),(6,1)],[(4,0),(4,0)],[(5,1),(5,1)]],[]],
                    [[(6,1)],[[(5,0),(5,0)],[(7,3),(7,3)],[(6,2),(6,2)],[(7,2),(7,2)]],[(5,0)]],
                    [[(7,3)],[[(6,2),(7,2)],[(8,4),(8,4)]],[(5,0),(6,1)]],
                    [[(9,4)],[[(6,2),(7,2)],[(8,4),(8,4)],[(8,3),(8,3)],[(10,5),(10,5)]],[(5,0),(6,1),(7,3)]],
                    [[(8,5)],[[(6,2),(7,2)],[(8,4),(7,4)],[(7,7),(9,6)]],[(5,0),(6,1),(7,3)]],
                    [[(9,6)],[[(9,5),(8,6)]],[(8,5)]],
                    [[(4,0)],[[(3,1),(3,1)],[(3,0),(4,1)]],[(5,0)]],
                    [[(3,1)],[[(4,0),(4,0)],[(2,2),(2,2)],[(0,0),(0,0)]],[(5,0),(4,0)]],
                    [[0,0],[[(3,0),(4,1)],[(2,1),(2,0)]],[(5,0),(4,0),(3,1)]],
                    ]

            if self.player_number ==1:
                for ord in lst:
                    b=False
                    try:
                        for i in ord[0]:
                            if state[i] == 2:
                                b=True
                                break
                        for i in ord[2]:
                            if state[i] != 1:
                                b=True
                                break
                    except:
                        continue
                    if b:
                        continue
                    counter=0
                    for group in ord[1]:
                        if state[group[0]]==2 and state[group[1]]==2:
                            break
                        if state[group[0]]==2 and state[group[1]]==0:
                            counter+=1
                        if state[group[1]]==2 and state[group[0]]==0:
                            counter+=1
                    if counter==2:
                        continue
                    for group in ord[1]:
                        if state[group[0]]==0 and state[group[1]]==2:
                            return group[0],[],0
                        if state[group[0]]==2 and state[group[1]]==0:
                            return group[1],[],0
                    for i in ord[0]:
                        if state[i] == 0:
                            print("move ",i)
                            return i, [], 0
            else:
                for ord in lst:
                    b=False
                    try:

                        for i in ord[0]:
                            print(state[i])
                            if state[i] == 1:
                                b=True
                                break
                        for i in ord[2]:
                            if state[i] != 2:
                                b=True
                                break
                    except:
                        continue
                    if b:
                        continue
                    counter=0
                    for group in ord[1]:
                        if state[group[0]]==1 and state[group[1]]==1:
                            break
                        if state[group[0]]==1 and state[group[1]]==0:
                            counter+=1
                        if state[group[1]]==1 and state[group[0]]==0:
                            counter+=1
                    if counter==2:
                        continue
                    for group in ord[1]:
                        if state[group[0]]==0 and state[group[1]]==1:
                            return group[0],[],0
                        if state[group[0]]==1 and state[group[1]]==0:
                            return group[1],[],0
                    for i in ord[0]:
                        if state[i] == 0:
                            print("move ",i)
                            return i, [], 0



        print("time for precomputation ",time.time()-time_current)
        start_time = time.time()
        iterations = 0
        # Additional logic for move selection

        time_remaining = fetch_remaining_time(self.timer, self.player_number)
        print("time remaining ",time_remaining)
        total_moves = 37//2
        if self.dim == 11:
            total_moves = 100//2
        time_given= min(time_remaining/total_moves * 2,time_remaining/10)
        move_played=player1
        if move_played/total_moves > 0.5:
            time_given = min(time_remaining/total_moves * 1.5,time_remaining/10)
        if move_played/total_moves > 0.75:
            time_given = time_remaining/total_moves
        print("time given to",self.player_number,time_given)
        while time.time() - start_time < min(20,time_given):  # Run for 10 seconds
            node = self.select(root)
            if node.untried_moves:
                node = self.expand(node)
            result = self.simulate(node)
            self.backpropagate(node, result)
            iterations += 1

        best_child = max(root.children, key=lambda c: c.visits)
        
        # Prepare detailed child state information
        child_info = []
        for child in root.children:
            child_info.append({
                'move': child.move,
                'visits': child.visits,
                'wins': child.wins,
                'win_rate': child.wins / child.visits if child.visits > 0 else 0
            })
        # print("ai iteration: ",iterations)
        # print("best move type ",type(best_child.move[0]))
        return best_child.move, child_info, iterations

    def select(self, node):
        while node.children:
            if node.untried_moves:
                return node
            node = self.uct_select(node)
        return node

    def expand(self, node):
        move = node.untried_moves.pop()
        new_state = node.state.copy()
        new_state[move] = node.player
        child_node = Node(new_state, parent=node, move=move, player=3-node.player)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):
        state = node.state.copy()
        current_player = node.player
        last_move = node.move
        valid_moves = get_valid_actions(state)
        random.shuffle(valid_moves)
        valid_moves.append(last_move)


        while True:
            if last_move:
                res = check_fast_win(state, last_move, 3 - current_player, self.neighbours, self.corners, self.edges)
                if res:
                    return 1 if 3 - current_player == self.player_number else 0

            valid_moves.pop()
            if not valid_moves:
                return 0.5  # Draw if no valid moves

            move = valid_moves[-1]
            state[move] = current_player
            last_move = move
            current_player = 3 - current_player  # Switch player

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.wins += result if node.player == self.opponent_number else 1 - result
            node = node.parent

    def uct_select(self, node):
        return max(node.children, key=lambda c: self.uct_value(node, c))

    def uct_value(self, parent, child):
        if child.visits == 0:
            return float('inf')
        return (child.wins / child.visits) + self.C * math.sqrt(math.log(parent.visits) / child.visits)

class AIPlayer:
    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'AI m3 with fast win'
        self.player_string = 'Player {}: AI m2 with fast win'.format(player_number)
        self.timer = timer
        self.mcts = MCTS(player_number, timer)

    def get_move(self, state: np.array) -> Tuple[int, int]:
        remaining_time = fetch_remaining_time(self.timer, self.player_number)
        # print(state)
        
        # Log remaining time
        with open(f"mcts_log_player{self.player_number}.txt", "a") as f:
            f.write(f"Remaining time: {remaining_time:.2f} seconds\n")
        
        move, child_info, iterations = self.mcts.get_move(state)
        
        child_info = sorted(child_info, key=lambda x: x['win_rate'], reverse=True)

        # Log child state information
        with open(f"fast_mcts_multirollout{self.player_number}.txt", "a") as f:
            f.write(f"Total Iterations: {iterations}\n")
            f.write("Child States Information:\n")
            for child in child_info[:5]:
                f.write(f"Move: {child['move']}, Visits: {child['visits']}, "
                        f"Wins: {child['wins']}, Win Rate: {child['win_rate']:.2f}\n")
            f.write("\n")
        
        # move = self.mcts.get_move(state)
        return move
    
