import random
import copy


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']
    depth_max = 2

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """


        score = float("-inf")
        move = []
        is_drop_phase = self.is_drop_phase(state)

        # successors of state
        successors = self.succ(state, self.my_piece)
        # loop through each successor
        for succ in successors:
            # the successor state
            state_copy = succ[2]
            possible_score = self.max_value(state_copy, float("-inf"), float("inf"), 0)

            # when the score of the state of the successor is maximum
            if possible_score > score:
                score = possible_score
                # store the row and column of piece to place
                new_move = (succ[3], succ[4])
                # if the piece moves since all the pieces are placed, 
                # store the new row and new column
                if not self.is_drop_phase(state):
                    curr = (succ[0], succ[1])
        # if the piece moved in the state, 
        # store the new row and new column and 
        # old row and old column
        if not is_drop_phase:
            move.append(new_move)
            move.append(curr)
        # else, store the row and column of the piece to place
        # in the board
        else:
            move.append(new_move)

        return move
    

    def succ(self, state, piece):
        succ = []

        if self.is_drop_phase(state):
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[row][col] = piece
                        succ.append([row, col, new_state, row, col])
        else:
            for row in range(5):
                for col in range(5):
                    if state[row][col] == piece:
                        # when the piece is not at first column,
                        # and its left location is empty
                        # if col!= 0 and state[row][col-1] == ' ':
                        if col - 1 >= 0 and state[row][col-1] == ' ':
                            new_row = row
                            new_col = col-1
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col])
                        # when the piece is not at the last column,
                        # and its right location is empty
                        # if col != 4 and state[row][col+1] == ' ':
                        if col + 1 < 5 and state[row][col+1] == ' ': 
                            new_row = row
                            new_col = col+1
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col])
                        # when the piece is not at the first row
                        # and its above location is empty
                        # if row != 0 and state[row-1][col] == ' ':
                        if row - 1 >= 0 and state[row-1][col] == ' ':
                            new_row = row-1
                            new_col = col
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col])
                        # when the piece is not at the last row
                        # and its below location is empty 
                        # if row != 4 and state[row+1][col] == ' ':      
                        if row + 1 < 5 and state[row+1][col] == ' ': 
                            new_row = row+1
                            new_col = col
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col])
                        # when the piece is not at the first row and first column
                        # and its above left diagonal location is empty    
                        # if row != 0 and col != 0 and state[row-1][col-1] == ' ':
                        if row - 1 >= 0 and col - 1 >= 0 and state[row-1][col-1] == ' ':
                            new_row = row-1
                            new_col = col-1
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col])
                        # when the piece is not at the first row and last column
                        # and its above right diagonal location is empty
                        # if row != 0 and col != 4 and state[row-1][col+1] == ' ':
                        if row - 1 >= 0 and col + 1 < 5 and state[row-1][col+1] == ' ':
                            new_row = row-1
                            new_col = col+1
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col])
                        # when the piece is not at the last row and first column
                        # and its below right diagonal location is empty
                        # if row != 4 and col != 0 and state[row+1][col-1] == ' ':
                        if row + 1 < 5 and col - 1 >= 0 and state[row+1][col-1] == ' ':
                            new_row = row-1
                            new_col = col-1
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col]) 
                        # when the piece is not at the last row and last column
                        # and its below left diagonal location is empty
                        # if row != 4 and col != 4 and state[row+1][col+1] == ' ':
                        if row + 1 < 5 and col + 1 < 5 and state[row+1][col+1] == ' ':
                            new_row = row+1
                            new_col = col+1
                            new_state = copy.deepcopy(state)
                            new_state[row][col] = ' '
                            new_state[new_row][new_col] = piece
                            succ.append([row, col, new_state, new_row, new_col])
        return succ                                                                                                                          
        

    def is_drop_phase(self, state):
        piece = 0
        for row in range(5):
            for col in range(5):
                if state[row][col] != ' ':
                    piece += 1
        # return true if the the piece placed in the board is under 8
        return piece < 8


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for col in range(2):
            for i in range(0,2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col+1] == state[i+2][col+2] == state[i+3][col+3]:
                    return 1 if state[i][col]==self.my_piece else -1
        # TODO: check / diagonal wins
        for col in range(2):
            for i in range(3,5):
                if state[i][col] != ' ' and state[i][col] == state[i-1][col+1] == state[i-2][col+2] == state[i-3][col+3]:
                    return 1 if state[i][col]==self.my_piece else -1
        # TODO: check box wins
        for col in range(4):
            for i in range(4):
                if state[i][col] != ' ' and state[i][col] == state[i][col+1] == state[i+1][col+1] == state[i+1][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        return 0 # no winner yet


    def heuristic_game_value(self, state):
        # if there is winner or loser, return the value
        if self.game_value(state) != 0:
            return self.game_value(state)

        player_score = 0
        opp_score = 0
        # calculate the player's heuristic score
        player_score = self.evaluate_heuristic_continuous(state, self.my_piece) 
        # calculate the opponent's heuristic score
        opp_score = self.evaluate_heuristic_continuous(state, self.opp)

        return player_score - opp_score
    

    def evaluate_heuristic_continuous(self, state, piece):

        weight =[
            [0.03, 0.05, 0.05, 0.05, 0.03],
            [0.05, 0.1, 0.1, 0.1, 0.05],
            [0.05, 0.1, 0.3, 0.1, 0.05],
            [0.05, 0.1, 0.1, 0.1, 0.05],
            [0.03, 0.05, 0.05, 0.05, 0.03],
        ]

        for row in range(5):
            for col in range(5):
                if state[row][col] == self.my_piece:
                    self_weight = weight[row][col]

        # check horizontal wins
        for col in range(2):
            for i in range(5):
                if state[i][col] == piece:
                    count = 0
                    for k in range(1, 4):
                        if state[i][col] == state[i][col+k]:
                            count+= 1
                        else:
                            break
                    if count >= 2:
                        return 0.75
                    if count == 1:
                        return 0.5
                    if count == 0:
                        return self_weight    

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] == piece:
                    count = 0
                    for k in range(1, 4):
                        if state[i][col] == state[i+k][col]:
                            count+= 1
                        else:
                            break
                    if count >= 2:
                        return 0.75
                    if count == 1:
                        return 0.5
                    if count == 0:
                        return self_weight
                    
        # check \ diagonal wins
        for col in range(2):
            for i in range(0,2):
                if state[i][col] == piece:
                    count = 0
                    for k in range(1, 4):
                        if state[i][col] == state[i+k][col+k]:
                            count+= 1
                        else:
                            break
                    if count >= 2:
                        return 0.75
                    if count == 1:
                        return 0.5
                    if count == 0:
                        return self_weight

        # check / diagonal wins
        for col in range(2):
            for i in range(3,5):
                if state[i][col] == piece:
                    count = 0
                    for k in range(1, 4):
                        if state[i][col] == state[i-k][col+k]:
                            count+= 1
                        else: 
                            break
                    if count >= 2:
                        return 0.75
                    if count == 1:
                        return 0.5
                    if count == 0:
                        return self_weight
        
        # check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] == piece:
                    count = 0
                    if state[row][col] == state[row+1][col]: count += 1
                    if state[row][col] == state[row][col+1]: count += 1
                    if state[row][col] == state[row+1][col+1]: count += 1
                    else:
                        break
                    if count >= 2:
                        return 0.75
                    if count == 1:
                        return 0.5
                    if count == 0:
                        return self_weight
                    
        return self_weight
    
    
    def max_value(self, state, alpha, beta, depth):
        # if the player wins, return 1, if the player loses, return -1, 
        # if the there is no winner, return 0
        game_value = self.game_value(state)
        # if there is winner
        if game_value == 1 or game_value == -1:
            # return the value when there is a winner or loser
            return game_value
        
        # if it reaches the depth limit return the heuristic value
        if depth == self.depth_max:
            return self.heuristic_game_value(state)
        
        successors = self.succ(state, self.my_piece)
        for successor in successors:
            if self.is_drop_phase(state):
                state_copy = successor[2]
            else:
                state_copy = successor[2]

            alpha = max(alpha, self.min_value(state_copy, alpha, beta, depth+1) )

            if alpha >= beta:
                return beta
            
        return alpha
        
    def min_value(self, state, alpha, beta, depth):
        # if the player wins, return 1, if the player loses, return -1, 
        # if the there is no winner, return 0
        game_value = self.game_value(state)
        # return the value when there is a winner or loser
        if game_value == 1 or game_value == -1:
            # return the value
            return game_value
        
        # if it reaches the depth limit return the heuristic value
        if depth == self.depth_max:
            return self.heuristic_game_value(state)
        
        successors = self.succ(state, self.my_piece)
        for successor in successors:
            if self.is_drop_phase(state):
                state_copy = successor[2]
            else:
                state_copy = successor[2]
            beta = min(beta, self.max_value(state_copy, alpha, beta, depth+1) )

            if beta <= alpha:
                return alpha
            
        return beta


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
