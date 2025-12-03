# CMPUT 455 Assignment 4 starter code (PoE2)
# Implement the specified commands to complete the assignment
# Full assignment specification on Canvas

import sys
import time
import random

class CommandInterface:
    # The following is already defined and does not need modification
    # However, you may change or add to this code as you see fit, e.g. adding class variables to init

    def __init__(self):
        # Define the string to function command mapping
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,   # init_game w h p s [board]
            "show"     : self.show,
            "timelimit": self.timelimit,   # timelimit seconds
            "genmove"  : self.genmove,     # see assignment spec
            "play"     : self.play, 
            "score"    : self.score
        }

        self.board = [[0]]
        self.to_play = 1
        self.handicap = 0.0
        self.score_cutoff = float("inf")
        self.time_limit = 1
        self._time_up = False
        self.move_count = 0
        
        # zobrist hashing
        self.zobrist_table = None
        self.zobrist_player = None
        self.current_hash = 0
        
        # transposition table and search state
        self._tt = {}
        self._pv_move = None
        self.killer_moves = {}
        self.history = {}

    # Convert a raw string to a command and a list of arguments
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
        ##Error handling is currently commented out to enable better error messages
        ##You may want to re-enable this for your submission in case there are unknown errors
        #try:
        return self.command_dict[command](args)
        #except Exception as e: 
        #    print("Command '" + s + "' failed with exception:", file=sys.stderr)
        #    print(e, file=sys.stderr)
        #    print("= -1\n")
        #    return False
        
    # Will continuously receive and execute commands
    # Commands should return True on success, and False on failure
    # Every command will print '= 1' or '= -1' at the end of execution to indicate success or failure respectively
    def main_loop(self):
        while True:
            s = input()
            if s.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(s):
                print("= 1\n")

    # List available commands
    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    # Helper function for command argument checking
    # Will make sure there are enough arguments, and that they are valid integers
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
    
    # Command functions needed for playing.
    # Feel free to modify them if needed, but keep their functionality intact.

    def init_zobrist(self):
        random.seed(42)
        self.zobrist_table = {}
        for y in range(self.height):
            for x in range(self.width):
                for player in [1, 2]:
                    self.zobrist_table[(x, y, player)] = random.getrandbits(64)
        self.zobrist_player = [random.getrandbits(64), random.getrandbits(64)]
        self.current_hash = self.zobrist_player[self.to_play - 1]

    # init_game w h p s
    def init_game(self, args):
        # Check arguments
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
        
        #Initialize game state
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
        self.move_count = 0
        self._pv_move = None
        self._tt = {}
        self.killer_moves = {}
        self.history = {}
        
        self.init_zobrist()
        return True

    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

    # Sets the timelimit for genmove, non-negative integer
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
        except ValueError:
            print("Illegal move: " + " ".join(args), file=sys.stderr)
            return False
        
        if not (0 <= x < self.width) or not (0 <= y < self.height) or self.board[y][x] != 0:
            print("Illegal move: " + " ".join(args), file=sys.stderr)
            return False
        
        if self.p1_score >= self.score_cutoff or self.p2_score >= self.score_cutoff:
            print("Illegal move: " + " ".join(args), "game ended.", file=sys.stderr)
            return False
        
        # put the piece onto the board
        self.make_move(x, y)

        return True

    def score(self, args):
        p1_score, p2_score = self.calculate_score()
        print(p1_score, p2_score)
        return True

    def _board_full(self):
        return self.move_count >= self.width * self.height

    def _get_empties(self):
        return self.width * self.height - self.move_count

    def _count_line(self, x, y, dx, dy, player):
        length = 0
        nx, ny = x + dx, y + dy
        while 0 <= nx < self.width and 0 <= ny < self.height and self.board[ny][nx] == player:
            length += 1
            nx += dx
            ny += dy
        return length

    def _line_potential(self, x, y, dx, dy, player):
        fwd = self._count_line(x, y, dx, dy, player)
        bwd = self._count_line(x, y, -dx, -dy, player)
        return fwd + bwd

    def _evaluate(self):
        p1_score, p2_score = self.calculate_score()
        base = p1_score - p2_score if self.to_play == 1 else p2_score - p1_score
        
        # threat analysis - look at line extension potential
        my_threat = 0
        opp_threat = 0
        opp = 3 - self.to_play
        
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] != 0:
                    continue
                for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    my_len = self._line_potential(x, y, dx, dy, self.to_play)
                    if my_len >= 2:
                        my_threat += my_len * my_len
                    opp_len = self._line_potential(x, y, dx, dy, opp)
                    if opp_len >= 2:
                        opp_threat += opp_len * opp_len
        
        return base + 0.12 * (my_threat - 0.9 * opp_threat)

    def _tactical_score(self, x, y):
        score = 0
        opp = 3 - self.to_play
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            my_len = self._line_potential(x, y, dx, dy, self.to_play)
            if my_len > 0:
                score += my_len * my_len * 2
            opp_len = self._line_potential(x, y, dx, dy, opp)
            if opp_len > 0:
                score += opp_len * opp_len
        return score

    def _count_adjacent(self, x, y):
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.board[ny][nx] != 0:
                        count += 1
        return count

    def _order_moves(self, moves, depth, tt_best):
        if not moves:
            return []
        
        ordered = []
        remaining = list(moves)
        
        # TT best move first
        if tt_best and tt_best in remaining:
            ordered.append(tt_best)
            remaining.remove(tt_best)
        
        # killer moves
        if depth in self.killer_moves:
            for killer in self.killer_moves[depth]:
                if killer in remaining:
                    ordered.append(killer)
                    remaining.remove(killer)
        
        # score the rest
        cx = self.width / 2.0
        cy = self.height / 2.0
        scored = []
        for move in remaining:
            x, y = move
            s = 0
            s += self.history.get(move, 0)
            s += self._tactical_score(x, y) * 500
            s += self._count_adjacent(x, y) * 200
            dist = abs(x - cx) + abs(y - cy)
            s += (10 - dist) * 50
            scored.append((s, move))
        
        scored.sort(reverse=True, key=lambda p: p[0])
        ordered.extend([m for _, m in scored])
        return ordered

    def _negamax(self, depth, alpha, beta, deadline):
        if time.perf_counter() >= deadline:
            self._time_up = True
            return self._evaluate(), None

        if self._board_full():
            return self._evaluate(), None

        if depth == 0:
            return self._evaluate(), None

        empties = self._get_empties()
        if empties <= 12:
            effective_depth = min(depth, empties)
        else:
            effective_depth = depth

        alpha_orig = alpha
        hash_key = self.current_hash
        tt_best = None
        
        if hash_key in self._tt:
            tt_depth, tt_val, tt_flag, tt_move = self._tt[hash_key]
            tt_best = tt_move
            if tt_depth >= effective_depth:
                if tt_flag == 'exact':
                    return tt_val, tt_move
                elif tt_flag == 'lower':
                    alpha = max(alpha, tt_val)
                elif tt_flag == 'upper':
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val, tt_move

        moves = self.get_moves()
        if not moves:
            return self._evaluate(), None

        moves = self._order_moves(moves, depth, tt_best)

        best = -float("inf")
        best_move = None

        for move in moves:
            x, y = move
            self.make_move(x, y)
            val, _ = self._negamax(effective_depth - 1, -beta, -alpha, deadline)
            val = -val
            self.undo_move(x, y)

            if self._time_up:
                return best if best != -float("inf") else 0.0, best_move

            if val > best:
                best = val
                best_move = move
            if val > alpha:
                alpha = val
            if alpha >= beta:
                # update killer moves
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, move)
                    if len(self.killer_moves[depth]) > 2:
                        self.killer_moves[depth].pop()
                # history heuristic
                bonus = depth * depth
                self.history[move] = self.history.get(move, 0) + bonus
                break

        if not self._time_up:
            if best <= alpha_orig:
                tt_flag = 'upper'
            elif best >= beta:
                tt_flag = 'lower'
            else:
                tt_flag = 'exact'
            self._tt[hash_key] = (effective_depth, best, tt_flag, best_move)

        return best, best_move

    def _root_search(self, max_depth, deadline, alpha, beta):
        moves = self.get_moves()
        if not moves:
            return None, 0.0

        hash_key = self.current_hash
        tt_best = None
        if hash_key in self._tt:
            tt_best = self._tt[hash_key][3]
        
        moves = self._order_moves(moves, 0, tt_best)
        
        if self._pv_move and self._pv_move in moves:
            moves.remove(self._pv_move)
            moves.insert(0, self._pv_move)

        best_move = None
        best_val = -float("inf")

        for move in moves:
            x, y = move
            self.make_move(x, y)
            val, _ = self._negamax(max_depth - 1, -beta, -alpha, deadline)
            val = -val
            self.undo_move(x, y)

            if self._time_up:
                break

            if val > best_val or best_move is None:
                best_val = val
                best_move = move

            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        return best_move, best_val
    
    # Optional helper functions that you may use or replace with your own.

    def get_moves(self):
        moves = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    moves.append((x, y))
        return moves

    def make_move(self, x, y):
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        self.board[y][x] = self.to_play
        self.current_hash ^= self.zobrist_table[(x, y, self.to_play)]
        self.move_count += 1
        self.to_play = 3 - self.to_play
        self.current_hash ^= self.zobrist_player[self.to_play - 1]

    def undo_move(self, x, y):
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        self.to_play = 3 - self.to_play
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        self.current_hash ^= self.zobrist_table[(x, y, self.to_play)]
        self.board[y][x] = 0
        self.move_count -= 1

    # Returns p1_score, p2_score
    def calculate_score(self):
        p1_score = 0
        p2_score = self.handicap

        # Progress from left-to-right, top-to-bottom
        # We define lines to start at the topmost (and for horizontal lines leftmost) point of that line
        # At each point, score the lines which start at that point
        # By only scoring the starting points of lines, we never score line subsets
        for y in range(self.height):
            for x in range(self.width):
                c = self.board[y][x]
                if c != 0:
                    lone_piece = True # Keep track of the special case of a lone piece
                    # Horizontal
                    hl = 1
                    if x == 0 or self.board[y][x-1] != c: #Check if this is the start of a horizontal line
                        x1 = x+1
                        while x1 < self.width and self.board[y][x1] == c: #Count to the end
                            hl += 1
                            x1 += 1
                    else:
                        lone_piece = False
                    # Vertical
                    vl = 1
                    if y == 0 or self.board[y-1][x] != c: #Check if this is the start of a vertical line
                        y1 = y+1
                        while y1 < self.height and self.board[y1][x] == c: #Count to the end
                            vl += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Diagonal
                    dl = 1
                    if y == 0 or x == 0 or self.board[y-1][x-1] != c: #Check if this is the start of a diagonal line
                        x1 = x+1
                        y1 = y+1
                        while x1 < self.width and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            dl += 1
                            x1 += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Anit-diagonal
                    al = 1
                    if y == 0 or x == self.width-1 or self.board[y-1][x+1] != c: #Check if this is the start of an anti-diagonal line
                        x1 = x-1
                        y1 = y+1
                        while x1 >= 0 and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            al += 1
                            x1 -= 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Add scores for found lines
                    for line_length in [hl, vl, dl, al]:
                        if line_length > 1:
                            if c == 1:
                                p1_score += 2 ** (line_length-1)
                            else:
                                p2_score += 2 ** (line_length-1)
                    # If all found lines are length 1, check if it is the special case of a lone piece
                    if hl == vl == dl == al == 1 and lone_piece:
                        if c == 1:
                            p1_score += 1
                        else:
                            p2_score += 1
        return p1_score, p2_score
    
    def get_relative_score(self):
        p1_score, p2_score = self.calculate_score()
        if p1_score >= self.score_cutoff or p2_score >= self.score_cutoff:
            if self.to_play == 1:
                return p1_score - p2_score
            else:
                return p2_score - p1_score
        else:
            # Check if the board is full
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] == 0:
                        return None
            if self.to_play == 1:
                return p1_score - p2_score
            else:
                return p2_score - p1_score

    # To implement for assignment 4.
    # Make sure you print a move within the specified time limit (1 second by default)
    # Print the x and y coordinates of your chosen move, space separated.
    def genmove(self, args):
        moves = self.get_moves()
        if not moves:
            print("0 0")
            return True

        start = time.perf_counter()
        deadline = start + self.time_limit * 0.85

        self.killer_moves = {}
        
        # fallback - pick center-ish move
        cx, cy = self.width // 2, self.height // 2
        best_move = min(moves, key=lambda m: abs(m[0] - cx) + abs(m[1] - cy))
        best_val = -float("inf")

        self._pv_move = None
        depth = 1
        
        while True:
            self._time_up = False
            
            # aspiration window for depth > 2
            if depth > 2 and best_val != -float("inf"):
                window = 60
                alpha = best_val - window
                beta = best_val + window
                move, val = self._root_search(depth, deadline, alpha, beta)
                
                # re-search if outside window
                if not self._time_up and (val <= alpha or val >= beta):
                    move, val = self._root_search(depth, deadline, -float("inf"), float("inf"))
            else:
                move, val = self._root_search(depth, deadline, -float("inf"), float("inf"))

            if self._time_up:
                break

            if move is not None:
                best_move = move
                best_val = val
                self._pv_move = move

            depth += 1
            
            elapsed = time.perf_counter() - start
            if elapsed > self.time_limit * 0.7:
                break
            
            if depth > self.width * self.height:
                break

        x, y = best_move
        self.make_move(x, y)
        print(x, y)
        return True

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
