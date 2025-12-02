# CMPUT 455 Assignment 4 starter code (PoE2)
# Implement the specified commands to complete the assignment
# Full assignment specification on Canvas

import sys
import time

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
        # Internal flag used by the search to stop when time is up
        self._time_up = False

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
    
    # -----------------------------
    # Search helpers for genmove()
    # -----------------------------

    def _board_full(self):
        """Return True if there are no empty points left."""
        for row in self.board:
            for v in row:
                if v == 0:
                    return False
        return True

    def _evaluate(self):
        """
        Heuristic evaluation: score difference from the perspective
        of the current player (self.to_play).
        """
        p1_score, p2_score = self.calculate_score()
        if self.to_play == 1:
            return p1_score - p2_score
        else:
            return p2_score - p1_score

    def _move_order_key(self, move):
        """
        Move ordering heuristic:
        - Prefer moves near the center
        - Prefer moves adjacent to existing stones
        """
        x, y = move

        # Center preference (negative squared distance from center)
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        center_score = -((x - cx) ** 2 + (y - cy) ** 2)

        # Adjacency score: count neighboring stones (8 directions)
        adj = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.board[ny][nx] != 0:
                        adj += 1

        # Adjacency dominates, center breaks ties
        return adj * 10.0 + center_score

    def _negamax(self, depth, alpha, beta, deadline):
        """
        Standard negamax with alpha-beta pruning.
        Values are always from the perspective of the player to move
        (self.to_play) at this node.
        """
        # Time check
        if time.perf_counter() >= deadline:
            self._time_up = True
            return self._evaluate()

        # Leaf node: depth limit or full board -> evaluate
        if depth == 0 or self._board_full():
            return self._evaluate()

        moves = self.get_moves()
        if not moves:
            # Safety: no legal moves, treat as leaf
            return self._evaluate()

        # Move ordering to improve alpha-beta pruning
        moves.sort(key=self._move_order_key, reverse=True)

        best = -float("inf")

        for (x, y) in moves:
            self.make_move(x, y)
            val = -self._negamax(depth - 1, -beta, -alpha, deadline)
            self.undo_move(x, y)

            if self._time_up:
                # Time is up: return the best value found so far (if any),
                # or a safe fallback of 0.
                return best if best != -float("inf") else 0.0

            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                # Beta cutoff
                break

        return best

    def _root_search(self, max_depth, deadline):
        """
        Root-level search: returns (best_move, best_value) for the given depth.
        """
        best_move = None
        best_val = -float("inf")

        moves = self.get_moves()
        if not moves:
            return None, 0.0

        # Order moves at root too
        moves.sort(key=self._move_order_key, reverse=True)

        alpha = -float("inf")
        beta = float("inf")

        for (x, y) in moves:
            self.make_move(x, y)
            val = -self._negamax(max_depth - 1, -beta, -alpha, deadline)
            self.undo_move(x, y)

            if self._time_up:
                # Time ran out during child search, stop and keep what we have
                break

            if val > best_val or best_move is None:
                best_val = val
                best_move = (x, y)

            if val > alpha:
                alpha = val

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
        self.board[y][x] = self.to_play
        if self.to_play == 1:
            self.to_play = 2
        else:
            self.to_play = 1

    def undo_move(self, x, y):
        self.board[y][x] = 0
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
        """
        Generate and play a move using iterative deepening alpha-beta search.
        Respects self.time_limit (seconds) with a small safety margin.
        """
        moves = self.get_moves()
        if not moves:
            # No legal moves (should only happen on full board)
            # Output something legal-ish just in case
            print("0 0")
            return True

        start = time.perf_counter()
        # Safety: don't use full timelimit, and handle t=0 gracefully
        effective_limit = max(0.001, float(self.time_limit))
        deadline = start + max(0.01, 0.9 * effective_limit)

        # Fallback: if search fails or times out very early, play the first move
        best_move = moves[0]
        best_val = -float("inf")

        depth = 1
        # Iterative deepening: increase depth until time runs out
        while True:
            self._time_up = False
            move, val = self._root_search(depth, deadline)

            # If we completed this depth within the time limit and found a move,
            # update our best answer and go deeper.
            if not self._time_up and move is not None:
                best_move, best_val = move, val
                depth += 1

                # In theory we can't search deeper than number of empty squares
                if depth > self.width * self.height:
                    break
            else:
                # Either time ran out or something went wrong at this depth
                break

        x, y = best_move
        # Play the move on our internal board
        self.make_move(x, y)
        # Print the coordinates as required: "x y"
        print(x, y)
        return True

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
