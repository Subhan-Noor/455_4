
import sys
import time
import math
import random


MCTS_EXPLORATION = 0.4  #UCB   constant : lower means more exploit and higher means explore
RAVE_CONSTANT = 300     #RAVE parameter
TIME_BUFFER = 0.05      #buffer 

class BoardUtils:
    """Centralized game  logic"""
    
    @staticmethod
    def get_legal_moves(board, width, height):
        #move ordering
        moves = []
        cx, cy = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                if board[y][x] == 0:
                    dist = abs(x - cx) + abs(y - cy)
                    moves.append(((x, y), dist))
        moves.sort(key=lambda k: k[1])
        return [m[0] for m in moves]

    @staticmethod
    def calculate_score(board, width, height, handicap):
        p1_score = 0
        p2_score = handicap
        


        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        
        for y in range(height):
            for x in range(width):
                c = board[y][x]
                if c == 0: continue
                
                is_lone = True
                
                for dx, dy in dirs:
                    #check backward
                    prev_x, prev_y = x - dx, y - dy
                    if 0 <= prev_x < width and 0 <= prev_y < height and board[prev_y][prev_x] == c:
                        is_lone = False
                        continue
                    
                    #count forward length
                    length = 1
                    curr_x, curr_y = x + dx, y + dy
                    while 0 <= curr_x < width and 0 <= curr_y < height and board[curr_y][curr_x] == c:
                        length += 1
                        curr_x += dx
                        curr_y += dy
                    
                    if length > 1:
                        is_lone = False
                        points = 2 ** (length - 1)
                        if c == 1: p1_score += points
                        else: p2_score += points
                
                #cannot  part of any line
                if is_lone:
                    has_neighbor = False
                    for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1), (x+1,y+1), (x-1,y-1), (x+1,y-1), (x-1,y+1)]:
                        if 0 <= nx < width and 0 <= ny < height and board[ny][nx] == c:
                            has_neighbor = True
                            break
                    if not has_neighbor:
                        if c == 1: p1_score += 1
                        else: p2_score += 1
                        
        return p1_score, p2_score

class MCTSNode:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move 
        self.children = {}  # move_tuple maps to MCTSNode
        
        self.visits = 0

        self.value = 0.0    
        

        self.rave_visits = 0

        self.rave_value = 0.0
        
        self.untried_moves = None
        self.player = 0 

class MCTS:
    def __init__(self, width, height, handicap):
        self.width = width
        self.height = height
        self.handicap = handicap
        self.root = MCTSNode()
        self.root.untried_moves = None 

    def advance_root(self, move):
        """Move the root down the tree if the node exists, else reset."""
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None #detach
        else:
            self.root = MCTSNode()
            self.root.untried_moves = None

    def run(self, board, player_color, time_limit):
        start_time = time.time()
        

        if self.root.untried_moves is None:
            self.root.untried_moves = BoardUtils.get_legal_moves(board, self.width, self.height)
            self.root.player = 3 - player_color 

        root_player = player_color 
        
        #now doing mcts
        loops = 0
        while time.time() - start_time < (time_limit - TIME_BUFFER):
            loops += 1
            node = self.root
            
            sim_board = [row[:] for row in board]
            
            while not node.untried_moves and node.children:
                node = self.select_child(node)

                sim_board[node.move[1]][node.move[0]] = node.player

            #expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                next_player = 3 - node.player 
                child = MCTSNode(parent=node, move=move)
                child.player = next_player
                
                sim_board[move[1]][move[0]] = child.player
                
                child.untried_moves = BoardUtils.get_legal_moves(sim_board, self.width, self.height)
                child.untried_moves = BoardUtils.get_legal_moves(sim_board, self.width, self.height)
                
                node.children[move] = child
                node = child
            
            #simulation
            amaf_moves = {1: set(), 2: set()}
            
            curr = node
            while curr.parent:
                amaf_moves[curr.player].add(curr.move)
                curr = curr.parent
            
            score_diff = self.simulate_heuristic(sim_board, 3 - node.player, amaf_moves)

            #backpropagation
            norm_score = max(-1.0, min(1.0, score_diff / 100.0))
            
            while node:
                node.visits += 1
                
                val_update = norm_score if self.root.player != 1 else norm_score 
                
                if node.player == 1:
                    node.value += norm_score
                elif node.player == 2:
                    node.value -= norm_score
                
                #RAVE update siblings if their move in AMAF
                if node.parent:
                    for sibling in node.parent.children.values():
                        if sibling.move in amaf_moves[sibling.player]:
                            sibling.rave_visits += 1
                            if sibling.player == 1:
                                sibling.rave_value += norm_score
                            else:
                                sibling.rave_value -= norm_score
                
                node = node.parent

        #final selection
        if not self.root.children:
            legal = BoardUtils.get_legal_moves(board, self.width, self.height)
            return legal[0] if legal else None
            
        best_move = max(self.root.children.items(), key=lambda i: i[1].visits)[0]
        return best_move

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        
        log_N = math.log(node.visits)
        
        for child in node.children.values():
            win_rate = child.value / child.visits
            explore = MCTS_EXPLORATION * math.sqrt(log_N / child.visits)
            
            #RAVE
            if child.rave_visits > 5: 
                beta = child.rave_visits / (child.rave_visits + child.visits + 1e-5 * child.visits * 0) #Simplified Beta
                beta = math.sqrt(RAVE_CONSTANT / (3 * node.visits + RAVE_CONSTANT))
                
                rave_rate = child.rave_value / child.rave_visits
                combined = (1 - beta) * win_rate + beta * rave_rate
            else:
                combined = win_rate
                
            ucb_val = combined + explore
            
            if ucb_val > best_score:
                best_score = ucb_val
                best_child = child
                
        return best_child

    def simulate_heuristic(self, board, turn, amaf_moves):
        empty = []
        for y in range(self.height):
            for x in range(self.width):
                if board[y][x] == 0: empty.append((x, y))
        
        random.shuffle(empty)
        
        while empty:
            candidates = empty[-4:] #4 randoms
            best_c = None
            best_h = -999
            
            for cx, cy in candidates:
                h_val = 0
                h_val -= (abs(cx - self.width//2) + abs(cy - self.height//2))
                has_adj = False
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = cx+dx, cy+dy
                    if 0<=nx<self.width and 0<=ny<self.height and board[ny][nx] != 0:
                        has_adj = True
                        break
                if has_adj: h_val += 2
                
                if h_val > best_h:
                    best_h = h_val
                    best_c = (cx, cy)
            
            move = best_c
            empty.remove(move)
            
            board[move[1]][move[0]] = turn
            amaf_moves[turn].add(move)
            turn = 3 - turn
            
        p1, p2 = BoardUtils.calculate_score(board, self.width, self.height, self.handicap)
        return p1 - p2

class CommandInterface:
    def __init__(self):
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,
            "show"     : self.show,
            "timelimit": self.timelimit,
            "genmove"  : self.genmove,
            "play"     : self.play, 
            "undo"     : self.undo,
            "legal"    : self.legal,
            "score"    : self.score,
            "winner"   : self.winner
        }
        self.board = [[0]]
        self.width = 7
        self.height = 7
        self.to_play = 1
        self.handicap = 0.0
        self.score_cutoff = float("inf")
        self.time_limit = 1
        self.history = []
        self.mcts = None #persist MCTS tree

    def process_command(self, s):
        s = s.lower().strip()
        if len(s) == 0: return True
        parts = s.split(" ")
        cmd = parts[0]
        args = [x for x in parts[1:] if len(x) > 0]
        if cmd not in self.command_dict:
            print("? Unknown command", file=sys.stderr)
            return False
        return self.command_dict[cmd](args)

    def main_loop(self):
        while True:
            try: line = input()
            except EOFError: break
            if line.strip() == "exit":
                print("= 1\n"); break
            if self.process_command(line):
                print("= 1\n")
            else:
                print("= -1\n")

    def arg_check(self, args, template):
        if len(args) < len(template.split(" ")): return False
        for i, a in enumerate(args):
            try: args[i] = int(a)
            except: 
                try: args[i] = float(a)
                except: return False
        return True



    def init_game(self, args):
        if not self.arg_check(args, "w h p s"): return False
        self.width, self.height, self.handicap, s = args
        self.score_cutoff = float("inf") if s == 0 else s
        self.board = [[0]*self.width for _ in range(self.height)]
        self.to_play = 1
        self.history = []

        self.mcts = MCTS(self.width, self.height, self.handicap)
        return True

    def timelimit(self, args):
        if not self.arg_check(args, "t"): return False
        self.time_limit = args[0]
        return True

    def genmove(self, args):
        if self.mcts is None:
            self.mcts = MCTS(self.width, self.height, self.handicap)
            
        move = self.mcts.run(self.board, self.to_play, self.time_limit)
        
        if move:
            self.make_move_internal(move[0], move[1])
            print(f"{move[0]} {move[1]}")
        else:
            pass 
        return True

    def play(self, args):
        if not self.arg_check(args, "x y"): return False
        x, y = args
        if not (0 <= x < self.width and 0 <= y < self.height) or self.board[y][x] != 0:
            return False
        self.make_move_internal(x, y)
        return True

    def make_move_internal(self, x, y):
        self.board[y][x] = self.to_play
        self.history.append((x, y))
        self.to_play = 3 - self.to_play

        if self.mcts:
            self.mcts.advance_root((x, y))

    def undo(self, args):
        if not self.history: return False
        x, y = self.history.pop()
        self.board[y][x] = 0
        self.to_play = 3 - self.to_play

        self.mcts = MCTS(self.width, self.height, self.handicap)
        return True

    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

    def score(self, args):
        s1, s2 = BoardUtils.calculate_score(self.board, self.width, self.height, self.handicap)
        print(s1, s2)
        return True

    def winner(self, args):
        s1, s2 = BoardUtils.calculate_score(self.board, self.width, self.height, self.handicap)
        if s1 > s2: print(1)
        elif s2 > s1: print(2)
        else: print(0)
        return True

    def legal(self, args):
        if not self.arg_check(args, "c r"): return False
        x, y = args
        if 0 <= x < self.width and 0 <= y < self.height and self.board[y][x] == 0:
            print("yes")
        else: print("no")
        return True

    def help(self, args):
        print("Available:", ", ".join(self.command_dict.keys()))
        return True

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
    