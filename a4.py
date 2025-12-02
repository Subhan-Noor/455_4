# CMPUT 455 Assignment 4 starter code (PoE2)
# Implement the specified commands to complete the assignment
# Full assignment specification on Canvas

import sys
import time
import random
from collections import defaultdict


class SearchTimeout(Exception):
    pass


class CommandInterface:
    # The following is already defined and does not need modification
    # However, you may change or add to this code as you see fit, e.g. adding class variables to init

    TT_EXACT = 0
    TT_LOWER = 1
    TT_UPPER = 2

    def __init__(self):
        # Define the string to function command mapping
        self.command_dict = {
            "help": self.help,
            "init_game": self.init_game,   # init_game w h p s [board]
            "show": self.show,
            "timelimit": self.timelimit,   # timelimit seconds
            "genmove": self.genmove,       # see assignment spec
            "play": self.play,
            "score": self.score
        }

        self.width = 1
        self.height = 1
        self.board = [[0]]
        self.to_play = 1
        self.handicap = 0.0
        self.score_cutoff = float("inf")
        self.time_limit = 1

        self.rand = random.Random(42)
        self.zobrist_table = []
        self.current_hash = 0
        self.pattern_dirs = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        self.tt_bits = 20
        self.tt_size = 1 << self.tt_bits
        self.transposition_table = [None] * self.tt_size
        self.history_table = defaultdict(int)
        self.killer_moves = [[None, None] for _ in range(64)]  # 2 killers per depth
        self.time_margin = 0.9
        self.search_deadline = 0.0

        self._init_zobrist_table()
        self.init_zobrist_hash()

    # Convert a raw string to a command and a list of arguments
    def process_command(self, s):
        s = s.lower().strip()
        if len(s) == 0:
            return True
        command = s.split(" ")[0]
        args = [x for x in s.split(" ")[1:] if len(x) > 0]
        if command not in self.command_dict:
            print("? Unknown command.\nType 'help' to list known commands.", file=sys.stderr, flush=True)
            return False
        try:
            return self.command_dict[command](args)
        except Exception as e: 
            print("Command '" + s + "' failed with exception:", file=sys.stderr, flush=True)
            print(e, file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return False

    # Will continuously receive and execute commands
    # Commands should return True on success, and False on failure
    # Every command will print '= 1' or '= -1' at the end of execution to indicate success or failure respectively
    def main_loop(self):
        while True:
            try:
                s = input()
            except EOFError:
                return True
            if s.split(" ")[0] == "exit":
                sys.stdout.write("= 1\n")
                sys.stdout.flush()
                return True
            try:
                result = self.process_command(s)
            except Exception:
                result = False
            if result:
                sys.stdout.write("= 1\n")
            else:
                sys.stdout.write("= -1\n")
            sys.stdout.flush()

    # List available commands
    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                sys.stdout.write(command + "\n")
        sys.stdout.write("exit\n")
        sys.stdout.flush()
        return True

    # Helper function for command argument checking
    # Will make sure there are enough arguments, and that they are valid integers
    def arg_check(self, args, template):
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr, flush=True)
            print("Recieved arguments: ", end="", file=sys.stderr)
            for a in args:
                print(a, end=" ", file=sys.stderr)
            print(file=sys.stderr, flush=True)
            return False
        for i, arg in enumerate(args):
            try:
                args[i] = int(arg)
            except ValueError:
                try:
                    args[i] = float(arg)
                except ValueError:
                    print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template, file=sys.stderr, flush=True)
                    return False
        return True
    
    # Command functions needed for playing.
    # Feel free to modify them if needed, but keep their functionality intact.

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
            print("Invalid board size:", w, h, file=sys.stderr, flush=True)
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
            self.board.append([0] * self.width)
        self.to_play = 1
        self.p1_score = 0
        self.p2_score = self.handicap

        self.transposition_table = [None] * self.tt_size
        self.history_table.clear()
        self.killer_moves = [[None, None] for _ in range(64)]
        self._init_zobrist_table()
        self.init_zobrist_hash()
        return True

    def show(self, args):
        for row in self.board:
            sys.stdout.write(" ".join(["_" if v == 0 else str(v) for v in row]) + "\n")
        sys.stdout.flush()
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
        except (ValueError, IndexError):
            return False

        if not (0 <= x < self.width) or not (0 <= y < self.height):
            return False
        if self.board[y][x] != 0:
            return False

        p1_score, p2_score = self.calculate_score()
        if p1_score >= self.score_cutoff or p2_score >= self.score_cutoff:
            return False
        
        try:
            self.make_move(x, y)
        except Exception:
            return False

        return True

    def score(self, args):
        p1_score, p2_score = self.calculate_score()
        sys.stdout.write(str(p1_score) + " " + str(p2_score) + "\n")
        sys.stdout.flush()
        return True
    
    # Optional helper functions that you may use or replace with your own.

    def get_moves(self):
        moves = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    moves.append((x, y))
        return moves

    def make_move(self, x, y):
        old_value = self.board[y][x]
        new_value = self.to_play
        self.board[y][x] = new_value
        self.update_hash(x, y, old_value, new_value)
        if self.to_play == 1:
            self.to_play = 2
        else:
            self.to_play = 1

    def undo_move(self, x, y):
        old_value = self.board[y][x]
        self.board[y][x] = 0
        self.update_hash(x, y, old_value, 0)
        if self.to_play == 1:
            self.to_play = 2
        else:
            self.to_play = 1

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
                    if x == 0 or self.board[y][x - 1] != c: #Check if this is the start of a horizontal line
                        x1 = x + 1
                        while x1 < self.width and self.board[y][x1] == c: #Count to the end
                            hl += 1
                            x1 += 1
                    else:
                        lone_piece = False
                    # Vertical
                    vl = 1
                    if y == 0 or self.board[y - 1][x] != c: #Check if this is the start of a vertical line
                        y1 = y + 1
                        while y1 < self.height and self.board[y1][x] == c: #Count to the end
                            vl += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Diagonal
                    dl = 1
                    if y == 0 or x == 0 or self.board[y - 1][x - 1] != c: #Check if this is the start of a diagonal line
                        x1 = x + 1
                        y1 = y + 1
                        while x1 < self.width and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            dl += 1
                            x1 += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Anit-diagonal
                    al = 1
                    if y == 0 or x == self.width - 1 or self.board[y - 1][x + 1] != c: #Check if this is the start of an anti-diagonal line
                        x1 = x - 1
                        y1 = y + 1
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
                                p1_score += 2 ** (line_length - 1)
                            else:
                                p2_score += 2 ** (line_length - 1)
                    # If all found lines are length 1, check if it is the special case of a lone piece
                    if hl == vl == dl == al == 1 and lone_piece:
                        if c == 1:
                            p1_score += 1
                        else:
                            p2_score += 1
        return p1_score, p2_score

    def get_relative_score(self):
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

    def genmove(self, args):
        moves = self.get_moves()
        if not moves:
            sys.stdout.write(str(self.width // 2) + " " + str(self.height // 2) + "\n")
            sys.stdout.flush()
            return True

        time_budget = max(0.2, self.time_limit * self.time_margin)
        self.history_table.clear()
        best_move = self.iterative_deepening_search(time_budget)
        if best_move is None:
            best_move = moves[0]
        self.make_move(*best_move)
        sys.stdout.write(str(best_move[0]) + " " + str(best_move[1]) + "\n")
        sys.stdout.flush()
        return True

    # --- Search helpers ---

    def _init_zobrist_table(self):
        self.rand.seed(42)
        self.zobrist_table = [
            [[self.rand.getrandbits(64) for _ in range(3)] for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def init_zobrist_hash(self):
        self.current_hash = 0
        if not self.zobrist_table:
            return
        for y in range(self.height):
            for x in range(self.width):
                cell = self.board[y][x]
                self.current_hash ^= self.zobrist_table[y][x][cell]

    def update_hash(self, x, y, old_value, new_value):
        if not self.zobrist_table:
            return
        if y < 0 or y >= len(self.zobrist_table) or x < 0 or x >= len(self.zobrist_table[0]):
            return
        if old_value < 0 or old_value >= 3 or new_value < 0 or new_value >= 3:
            return
        self.current_hash ^= self.zobrist_table[y][x][old_value]
        self.current_hash ^= self.zobrist_table[y][x][new_value]

    def _ensure_time(self):
        if time.perf_counter() >= self.search_deadline:
            raise SearchTimeout()

    def iterative_deepening_search(self, time_budget):
        start = time.perf_counter()
        self.search_deadline = start + time_budget
        pv_move = None
        root_moves = self.order_moves(self.get_moves(), None, 0, pv_move)
        if not root_moves:
            return None
        best_move = root_moves[0]
        depth = 1
        last_depth_time = 0.0
        while True:
            depth_start = time.perf_counter()
            try:
                value, move = self.negamax_alpha_beta(0, -float('inf'), float('inf'), depth)
            except SearchTimeout:
                break
            depth_time = time.perf_counter() - depth_start
            last_depth_time = depth_time
            if move is not None:
                best_move = move
                pv_move = move
            depth += 1
            if time.perf_counter() + max(0.01, last_depth_time * 1.75) > self.search_deadline:
                break
        return best_move

    def negamax_alpha_beta(self, depth, alpha, beta, max_depth):
        self._ensure_time()
        terminal, rel_score = self.get_relative_score()
        if terminal:
            return rel_score, None
        if depth >= max_depth:
            return self.evaluate_position(), None

        remaining = max_depth - depth
        entry = self._probe_tt()
        tt_move = entry[4] if entry else None
        if entry and entry[1] >= remaining:
            val = entry[2]
            f = entry[3]
            if f == self.TT_EXACT:
                return val, tt_move
            if f == self.TT_LOWER and val >= beta:
                return val, tt_move
            if f == self.TT_UPPER and val <= alpha:
                return val, tt_move

        moves = self.order_moves(self.get_moves(), tt_move, depth)
        if not moves:
            return rel_score, None

        best_move = None
        best_value = -float('inf')
        alpha_orig = alpha

        for move in moves:
            self.make_move(*move)
            try:
                child_value, _ = self.negamax_alpha_beta(depth + 1, -beta, -alpha, max_depth)
            finally:
                self.undo_move(*move)

            value = -child_value
            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                bonus = remaining * remaining if remaining > 0 else 1
                self.history_table[move] += bonus
                # Store killer move
                if depth < len(self.killer_moves):
                    if self.killer_moves[depth][0] != move:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                break

        if best_move is None:
            return rel_score, None

        if best_value <= alpha_orig:
            f = self.TT_UPPER
        elif best_value >= beta:
            f = self.TT_LOWER
        else:
            f = self.TT_EXACT
        self.tt_store(remaining, best_value, f, best_move)
        return best_value, best_move

    def _probe_tt(self):
        if self.tt_size == 0:
            return None
        idx = self.current_hash & (self.tt_size - 1)
        entry = self.transposition_table[idx]
        if entry and entry[0] == self.current_hash:
            return entry
        return None

    def tt_store(self, depth, value, flag, best_move):
        if self.tt_size == 0:
            return
        idx = self.current_hash & (self.tt_size - 1)
        self.transposition_table[idx] = (self.current_hash, depth, value, flag, best_move)

    def order_moves(self, moves, tt_move, depth=0, pv_move=None):
        if not moves:
            return []
        ordered = []
        seen_tt = False
        seen_pv = False
        k1, k2 = None, None
        if depth < len(self.killer_moves):
            k1, k2 = self.killer_moves[depth]
        
        # Only build groups at shallow depths (cheap and effective for ordering)
        use_merge = depth == 0
        group_map = group_sizes = group_owner = None
        if use_merge:
            group_map, group_sizes, group_owner = self._build_groups()
        
        for move in moves:
            score = 0
            
            # PV move gets highest priority
            if pv_move and move == pv_move and not seen_pv:
                score = 10000
                seen_pv = True
            elif tt_move and move == tt_move and not seen_tt:
                score = 8000
                seen_tt = True
            elif move == k1:
                score = 5000
            elif move == k2:
                score = 3000
            else:
                score = self.heuristic_move_score(move) + self.history_table[move]
                if use_merge and group_map is not None:
                    score += self._static_merge_score(move, group_map, group_sizes, group_owner)
            
            ordered.append((score, move))
        ordered.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in ordered]

    def heuristic_move_score(self, move):
        x, y = move
        p = self.to_play
        opp = 2 if p == 1 else 1
        
        # Center preference (important for opening)
        cx, cy = self.width // 2, self.height // 2
        center_bonus = (self.width - abs(x - cx)) + (self.height - abs(y - cy))
        
        # Count all neighbors
        my_adj = 0
        opp_adj = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.board[ny][nx] == p:
                        my_adj += 1
                    elif self.board[ny][nx] == opp:
                        opp_adj += 1
        
        # Balanced scoring
        score = center_bonus * 3
        score += my_adj * 8
        score += opp_adj * 4
        
        return score

    def _static_merge_score(self, move, group_map, group_sizes, group_owner):
        x, y = move
        if self.board[y][x] != 0:
            return 0
        p = self.to_play
        opp = 2 if p == 1 else 1
        my_groups = set()
        opp_groups = set()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                gid = group_map[ny][nx]
                if gid == -1:
                    continue
                owner = group_owner[gid]
                if owner == p:
                    my_groups.add(gid)
                elif owner == opp:
                    opp_groups.add(gid)
        bonus = 0
        if my_groups:
            bonus += 0.5 * self._merge_gain_from_groups(my_groups, group_sizes)
        if opp_groups:
            bonus += 0.6 * self._merge_gain_from_groups(opp_groups, group_sizes)
        return bonus

    # --- Evaluation ---

    def evaluate_position(self):
        p1_score, p2_score = self.calculate_score()
        if self.to_play == 1:
            base = p1_score - p2_score  # same as before
        else:
            # normalize away some of the handicap so P2 stays aggressive
            base = (p2_score - p1_score) - 3.0
        potential = self.estimate_potential_score()
        pattern = self.pattern_eval()
        base + 0.3*potential + 0.5*pattern

    def calc_merge_potential(self, group_map, group_sizes, group_owner):
        """Evaluate merge opportunities using exact exponential gains."""
        p = self.to_play
        opp = 2 if p == 1 else 1
        my_merge = 0
        opp_merge = 0
        
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] != 0:
                    continue
                my_groups = set()
                opp_groups = set()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        gid = group_map[ny][nx]
                        if gid == -1:
                            continue
                        owner = group_owner[gid]
                        if owner == p:
                            my_groups.add(gid)
                        elif owner == opp:
                            opp_groups.add(gid)
                if len(my_groups) >= 1:
                    my_merge += self._merge_gain_from_groups(my_groups, group_sizes)
                if len(opp_groups) >= 1:
                    opp_merge += self._merge_gain_from_groups(opp_groups, group_sizes)
        
        return my_merge - 1.0 * opp_merge

    def _merge_gain_from_groups(self, groups, group_sizes):
        merged_size = 1  # include the new stone
        current_score = 0
        for gid in groups:
            size = group_sizes.get(gid, 0)
            merged_size += size
            current_score += self._line_score(size)
        return self._line_score(merged_size) - current_score

    def estimate_potential_score(self):
        p = self.to_play
        my_pot = self._potential_for_player(p)
        other_p = 2 if p == 1 else 1
        other_pot = self._potential_for_player(other_p)
        return 0.5 * (my_pot - other_pot)

    def _potential_for_player(self, player):
        total = 0.0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == player:
                    total += self._count_adjacent_empty(x, y)
        return total

    def _count_adjacent_empty(self, x, y):
        empty = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.board[ny][nx] == 0:
                        empty += 1
        return empty

    def _mobility_score(self):
        p = self.to_play
        my_front = self._frontier_for_player(p)
        other = 2 if p == 1 else 1
        other_front = self._frontier_for_player(other)
        return my_front - other_front

    def _frontier_for_player(self, player):
        frontier = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] != 0:
                    continue
                has_neighbor = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.board[ny][nx] == player:
                                has_neighbor = True
                                break
                    if has_neighbor:
                        break
                if has_neighbor:
                    frontier += 1
        return frontier

    def pattern_eval(self):
        p = self.to_play
        my_score = self._pattern_score_for(p)
        other = 2 if p == 1 else 1
        other_score = self._pattern_score_for(other)
        return my_score - 0.8 * other_score

    def _pattern_score_for(self, player):
        score = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == player:
                    for dx, dy in self.pattern_dirs:
                        score += self._score_line(x, y, dx, dy, player)
        return score

    def _build_groups(self):
        group_map = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        group_sizes = {}
        group_owner = {}
        gid = 0
        for y in range(self.height):
            for x in range(self.width):
                player = self.board[y][x]
                if player == 0 or group_map[y][x] != -1:
                    continue
                size = self._flood_fill_group(x, y, player, gid, group_map)
                group_sizes[gid] = size
                group_owner[gid] = player
                gid += 1
        return group_map, group_sizes, group_owner

    def _flood_fill_group(self, x, y, player, gid, group_map):
        stack = [(x, y)]
        group_map[y][x] = gid
        size = 0
        while stack:
            cx, cy = stack.pop()
            size += 1
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.board[ny][nx] == player and group_map[ny][nx] == -1:
                        group_map[ny][nx] = gid
                        stack.append((nx, ny))
        return size

    def _line_score(self, length):
        if length <= 0:
            return 0
        return 1 << (length - 1)

    def _score_line(self, x, y, dx, dy, player):
        length = 1
        open_ends = 0
        nx, ny = x + dx, y + dy
        while self._on_board(nx, ny) and self.board[ny][nx] == player:
            length += 1
            nx += dx
            ny += dy
        if self._on_board(nx, ny) and self.board[ny][nx] == 0:
            open_ends += 1
        nx, ny = x - dx, y - dy
        while self._on_board(nx, ny) and self.board[ny][nx] == player:
            length += 1
            nx -= dx
            ny -= dy
        if self._on_board(nx, ny) and self.board[ny][nx] == 0:
            open_ends += 1
        if length >= 4:
            return 12 * length + 4 * open_ends
        if length == 3:
            return 8 + 6 * open_ends
        if length == 2:
            return 2 + 2 * open_ends
        return 0

    def _on_board(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    # To implement for assignment 4.
    # Make sure you print a move within the specified time limit (1 second by default)
    # Print the x and y coordinates of your chosen move, space separated.


if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
