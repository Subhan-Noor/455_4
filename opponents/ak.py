
# CMPUT 455 Assignment 4 - Highly Optimized PoE2 Player
import sys
import signal
import time
import random

class CommandInterface:
    def __init__(self):
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,
            "show"     : self.show,
            "timelimit": self.timelimit,
            "genmove"  : self.genmove,
            "play"     : self.play, 
            "score"    : self.score
        }

        self.board = [[0]]
        self.to_play = 1
        self.handicap = 0.0
        self.score_cutoff = float("inf")
        self.time_limit = 1
        self.tt = {}
        self.move_ordering_history = {}
        self.killer_moves = {}
        self.zobrist_table = None

    def process_command(self, s):
        s = s.lower().strip()
        if len(s) == 0:
            return True
        command = s.split(" ")[0]
        args = [x for x in s.split(" ")[1:] if len(x) > 0]
        if command not in self.command_dict:
            print("? Uknown command.\nType 'help' to list known commands.", file=sys.stderr)
            print("= -1\n")
            return False
        return self.command_dict[command](args)
        
    def main_loop(self):
        while True:
            s = input()
            if s.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(s):
                print("= 1\n")

    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    def arg_check(self, args, template):
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            print("Recieved arguments: ", end="", file=sys.stderr)
            for a in args:
                print(a, end=" ", file=sys.stderr)
            print(file=sys.stderr)
            return False
        for i, arg in enumerate(args):
            try:
                args[i] = int(arg)
            except ValueError:
                try:
                    args[i] = float(arg)
                except ValueError:
                    print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template, file=sys.stderr)
                    return False
        return True

    def init_zobrist(self):
        """Initialize Zobrist hashing table"""
        random.seed(42)  # Fixed seed for reproducibility
        self.zobrist_table = {}
        for y in range(self.height):
            for x in range(self.width):
                for player in [1, 2]:
                    self.zobrist_table[(x, y, player)] = random.getrandbits(64)
        self.zobrist_player = [random.getrandbits(64), random.getrandbits(64)]

    def compute_zobrist_hash(self):
        """Compute Zobrist hash for current board state"""
        h = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] != 0:
                    h ^= self.zobrist_table[(x, y, self.board[y][x])]
        h ^= self.zobrist_player[self.to_play - 1]
        return h

    def init_game(self, args):
        if len(args) > 4:
            self.board_str = args.pop()
        else:
            self.board_str = ""
        if not self.arg_check(args, "w h p s"):
            return False
        w, h, p, s = args
        if not (1 <= w <= 20 and 1 <= h <= 20):
            print("Invalid board size:", w, h, file=sys.stderr)
            return False
        
        self.width = w
        self.height = h
        self.handicap = p
        if s == 0:
            self.score_cutoff = float("inf")
        else:
            self.score_cutoff = s
        
        self.board = []
        for r in range(self.height):
            self.board.append([0]*self.width)
        self.to_play = 1
        self.p1_score = 0
        self.p2_score = self.handicap
        self.tt = {}
        self.move_ordering_history = {}
        self.killer_moves = {}
        
        # Initialize Zobrist hashing
        self.init_zobrist()
        self.current_hash = self.compute_zobrist_hash()
        
        return True

    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

    def timelimit(self, args):
        if not self.arg_check(args, "t"):
            return False
        self.time_limit = int(args[0])
        return True

    def play(self, args):
        if not self.arg_check(args, "x y"):
            return False
        
        try:
            x = int(args[0])
            y = int(args[1])
        except (ValueError, TypeError):
            print("Illegal move: " + " ".join([str(a) for a in args]), file=sys.stderr)
            return False
        
        if not (0 <= x < self.width) or not (0 <= y < self.height) or self.board[y][x] != 0:
            print("Illegal move: " + " ".join([str(a) for a in args]), file=sys.stderr)
            return False
        
        if self.p1_score >= self.score_cutoff or self.p2_score >= self.score_cutoff:
            print("Illegal move: " + " ".join([str(a) for a in args]), "game ended.", file=sys.stderr)
            return False
        
        self.make_move(x, y)
        return True

    def score(self, args):
        p1_score, p2_score = self.calculate_score()
        print(p1_score, p2_score)
        return True

    def get_moves(self):
        """Get all legal moves, cached for efficiency"""
        moves = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    moves.append((x, y))
        return moves

    def make_move(self, x, y):
        """Make a move and update Zobrist hash"""
        self.board[y][x] = self.to_play
        self.current_hash ^= self.zobrist_table[(x, y, self.to_play)]
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        if self.to_play == 1:
            self.to_play = 2
        else:
            self.to_play = 1
        self.current_hash ^= self.zobrist_player[self.to_play - 1]

    def undo_move(self, x, y):
        """Undo a move and restore Zobrist hash"""
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        if self.to_play == 1:
            self.to_play = 2
        else:
            self.to_play = 1
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        self.current_hash ^= self.zobrist_table[(x, y, self.to_play)]
        self.board[y][x] = 0

    def calculate_score(self):
        """Calculate scores for both players"""
        p1_score = 0
        p2_score = self.handicap

        for y in range(self.height):
            for x in range(self.width):
                c = self.board[y][x]
                if c != 0:
                    lone_piece = True
                    # Horizontal
                    hl = 1
                    if x == 0 or self.board[y][x-1] != c:
                        x1 = x+1
                        while x1 < self.width and self.board[y][x1] == c:
                            hl += 1
                            x1 += 1
                    else:
                        lone_piece = False
                    # Vertical
                    vl = 1
                    if y == 0 or self.board[y-1][x] != c:
                        y1 = y+1
                        while y1 < self.height and self.board[y1][x] == c:
                            vl += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Diagonal
                    dl = 1
                    if y == 0 or x == 0 or self.board[y-1][x-1] != c:
                        x1 = x+1
                        y1 = y+1
                        while x1 < self.width and y1 < self.height and self.board[y1][x1] == c:
                            dl += 1
                            x1 += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Anti-diagonal
                    al = 1
                    if y == 0 or x == self.width-1 or self.board[y-1][x+1] != c:
                        x1 = x-1
                        y1 = y+1
                        while x1 >= 0 and y1 < self.height and self.board[y1][x1] == c:
                            al += 1
                            x1 -= 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Add scores
                    for line_length in [hl, vl, dl, al]:
                        if line_length > 1:
                            if c == 1:
                                p1_score += 2 ** (line_length-1)
                            else:
                                p2_score += 2 ** (line_length-1)
                    if hl == vl == dl == al == 1 and lone_piece:
                        if c == 1:
                            p1_score += 1
                        else:
                            p2_score += 1
        return p1_score, p2_score
    
    def evaluate_threats(self):
        """Enhanced evaluation considering line potential and threats"""
        p1_score, p2_score = self.calculate_score()
        base_score = p1_score - p2_score if self.to_play == 1 else p2_score - p1_score
        
        # Add bonus for line potential
        threat_score = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    # Check if this empty square extends existing lines
                    for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
                        for player in [1, 2]:
                            line_len = 1
                            # Check one direction
                            nx, ny = x + dx, y + dy
                            while 0 <= nx < self.width and 0 <= ny < self.height and self.board[ny][nx] == player:
                                line_len += 1
                                nx += dx
                                ny += dy
                            # Check opposite direction
                            nx, ny = x - dx, y - dy
                            while 0 <= nx < self.width and 0 <= ny < self.height and self.board[ny][nx] == player:
                                line_len += 1
                                nx -= dx
                                ny -= dy
                            
                            if line_len >= 2:
                                potential = line_len * line_len
                                if (player == 1 and self.to_play == 1) or (player == 2 and self.to_play == 2):
                                    threat_score += potential
                                else:
                                    threat_score -= potential
        
        return base_score + threat_score * 0.1

    def get_relative_score(self):
        """Get relative score for current player"""
        p1_score, p2_score = self.calculate_score()
        if self.to_play == 1:
            score = p1_score - p2_score
        else:
            score = p2_score - p1_score
        if p1_score >= self.score_cutoff or p2_score >= self.score_cutoff:
            return True, score
        else:
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] == 0:
                        return False, score
            return True, score

    def order_moves(self, moves, depth, best_move):
        """Advanced move ordering with multiple heuristics"""
        if not moves:
            return []
        
        ordered = []
        
        # 1. Best move from TT
        if best_move and best_move in moves:
            ordered.append(best_move)
            moves = [m for m in moves if m != best_move]
        
        # 2. Killer moves
        if depth in self.killer_moves:
            for killer in self.killer_moves[depth]:
                if killer in moves:
                    ordered.append(killer)
                    moves = [m for m in moves if m != killer]
        
        # 3. Score remaining moves
        scored_moves = []
        center_x, center_y = self.width // 2, self.height // 2
        
        for move in moves:
            score = 0
            x, y = move
            
            # History heuristic
            score += self.move_ordering_history.get(move, 0)
            
            # Center proximity (center is often strategic)
            dist = abs(x - center_x) + abs(y - center_y)
            score += (10 - dist) * 100
            
            # Adjacent to existing pieces (connectivity bonus)
            adjacent_count = 0
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.board[ny][nx] != 0:
                    adjacent_count += 1
            score += adjacent_count * 200
            
            # Line extension potential
            for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
                for player in [self.to_play]:
                    line_len = 0
                    # Check both directions
                    for direction in [1, -1]:
                        nx, ny = x + dx * direction, y + dy * direction
                        while 0 <= nx < self.width and 0 <= ny < self.height and self.board[ny][nx] == player:
                            line_len += 1
                            nx += dx * direction
                            ny += dy * direction
                    if line_len > 0:
                        score += (line_len * line_len) * 500
            
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True)
        ordered.extend([m for _, m in scored_moves])
        return ordered

    def negamax_alpha_beta(self, alpha, beta, depth, max_depth):
        """Negamax with alpha-beta pruning and enhancements"""
        hash_key = self.current_hash
        
        # Check TT with depth verification
        if hash_key in self.tt:
            tt_value, tt_flag, tt_best_move, tt_depth = self.tt[hash_key]
            if tt_depth >= max_depth - depth:
                if tt_flag == 'exact':
                    return tt_value, True, tt_best_move
                elif tt_flag == 'lower' and tt_value >= beta:
                    return tt_value, True, tt_best_move
                elif tt_flag == 'upper' and tt_value <= alpha:
                    return tt_value, True, tt_best_move
        
        # Check terminal
        terminal, score = self.get_relative_score()
        if terminal:
            # Bonus for winning quickly
            if score > 0:
                score += 1000 - depth
            elif score < 0:
                score -= 1000 - depth
            self.tt[hash_key] = (score, 'exact', None, max_depth - depth)
            return score, True, None
        
        # Depth limit - use enhanced evaluation
        if depth >= max_depth:
            eval_score = self.evaluate_threats()
            self.tt[hash_key] = (eval_score, 'exact', None, 0)
            return eval_score, False, None
        
        # Get and order moves
        moves = self.get_moves()
        tt_best = self.tt.get(hash_key, (None, None, None, None))[2]
        moves = self.order_moves(moves, depth, tt_best)
        
        if not moves:
            return 0, True, None
        
        value = -float('inf')
        best_found_move = None
        valid_result = True
        original_alpha = alpha
        
        for move in moves:
            self.make_move(*move)
            child_value, valid_child, _ = self.negamax_alpha_beta(-beta, -alpha, depth + 1, max_depth)
            self.undo_move(*move)
            
            child_value = -child_value
            
            if child_value > value:
                value = child_value
                best_found_move = move
            
            valid_result = valid_result and valid_child
            
            alpha = max(alpha, value)
            
            if alpha >= beta:
                # Beta cutoff - update killer moves and history
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, move)
                    if len(self.killer_moves[depth]) > 2:
                        self.killer_moves[depth].pop()
                
                self.move_ordering_history[move] = self.move_ordering_history.get(move, 0) + (max_depth - depth) ** 2
                
                # Store lower bound
                self.tt[hash_key] = (value, 'lower', best_found_move, max_depth - depth)
                return value, valid_result, best_found_move
        
        # Store in TT
        if value <= original_alpha:
            flag = 'upper'
        elif value >= beta:
            flag = 'lower'
        else:
            flag = 'exact'
        
        self.tt[hash_key] = (value, flag, best_found_move, max_depth - depth)
        return value, valid_result, best_found_move

    def genmove(self, args):
        """Generate best move with iterative deepening"""
        start_time = time.time()
        
        max_depth = 1
        best_move = None
        best_value = -float('inf')
        
        # Clear killer moves for new search
        self.killer_moves = {}
        
        # Get initial move as fallback
        moves = self.get_moves()
        if moves:
            center_x, center_y = self.width // 2, self.height // 2
            best_move = min(moves, key=lambda m: abs(m[0]-center_x) + abs(m[1]-center_y))
        
        try:
            while max_depth <= 20:  # Reasonable depth limit
                # Check time before starting new depth
                elapsed = time.time() - start_time
                if elapsed > self.time_limit * 0.7:
                    break
                
                # Aspiration window search for depths > 2
                if max_depth > 2 and best_value != -float('inf'):
                    window = 50
                    alpha = best_value - window
                    beta = best_value + window
                    
                    value, valid, move = self.negamax_alpha_beta(alpha, beta, 0, max_depth)
                    
                    # Research if outside window
                    if value <= alpha or value >= beta:
                        value, valid, move = self.negamax_alpha_beta(-float('inf'), float('inf'), 0, max_depth)
                else:
                    value, valid, move = self.negamax_alpha_beta(-float('inf'), float('inf'), 0, max_depth)
                
                # Check time after search
                elapsed = time.time() - start_time
                if elapsed > self.time_limit * 0.8:
                    # Use result only if we finished in time
                    if move is not None:
                        best_move = move
                        best_value = value
                    break
                
                if move is not None:
                    best_move = move
                    best_value = value
                
                # Stop if we solved the position
                if valid:
                    break
                
                max_depth += 1
                    
        except Exception as e:
            print(f"Error in search: {e}", file=sys.stderr)
        
        # Ensure we have a valid move
        if best_move is None:
            moves = self.get_moves()
            if moves:
                center_x, center_y = self.width // 2, self.height // 2
                best_move = min(moves, key=lambda m: abs(m[0]-center_x) + abs(m[1]-center_y))
        
        if best_move:
            self.make_move(*best_move)
            print(best_move[0], best_move[1])
        else:
            print("0 0")  # Emergency fallback
        
        return True

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()