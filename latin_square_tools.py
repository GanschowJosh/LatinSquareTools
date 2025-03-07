from collections import defaultdict
from itertools import combinations, permutations
import random


################################################################ Hamiltonicity Tools ###########################################################################

def is_hamiltonian_cycle(graph):
    """
    Checks if a graph is a single hamiltonian cycle
    Parameters:
        graph (dict): A dictionary where the keys are the nodes and the values are the nodes connected to the key node
    Returns:
        bool: True if the graph is a single hamiltonian cycle, False otherwise
    """
    length = len(graph)
    seen = set()
    current_node = 0
    while current_node not in seen:
        if len(seen) == length:
            break
        seen.add(current_node)
        current_node = (
            graph[current_node][0]
            if graph[current_node][0] not in seen
            else graph[current_node][1]
        )
    if len(seen) != length:
        return False

    return True


def map_symbol_pairs_to_graph(pair, matrix):
    """
    Generates a graph from a pair of symbols
    Parameters:
        pair (tuple): A pair of symbols
        matrix (list): A list of lists representing a matrix
    Returns:
        dict: A dictionary where the keys are the nodes and the values are the nodes connected to the key node
    """
    graph = defaultdict(list)
    for row in matrix:
        current_edge = tuple()
        for i in range(len(row)):
            if row[i] == pair[0] or row[i] == pair[1]:
                current_edge += (i,)
            if len(current_edge) == 2:
                graph[current_edge[0]].append(current_edge[1])
                graph[current_edge[1]].append(current_edge[0])
                current_edge = tuple()
                break
    return graph


def ce_hamiltonian_checker(matrix):
    """
    checks if a matrix is consecutive entry hamiltonian, i.e., all consecutive pairs of symbols form a hamiltonian cycle

    Parameters:
        matrix (list): A list of lists representing a matrix
    Returns:
        bool: True if the matrix is CE-hamiltonian, False otherwise
    """
    length = len(matrix)
    consecutive_pairs = [(i, i + 1) for i in range(length - 1)] + [(length - 1, 0)]
    current_row = 0
    for pair in consecutive_pairs:
        graph = map_symbol_pairs_to_graph(pair, matrix)

        # checking hamiltonian cycle
        if not is_hamiltonian_cycle(graph):
            print(f"=============\nMatrix is not CE-hamiltonian, Problem pair: {pair}")
            return False

        current_row += 1
    return True


def cr_hamiltonian_checker(matrix):
    """
    Checks if a given Latin Square is consecutive-row hamiltonian, i.e., all consecutive rows are hamiltonian cycles
    Parameters:
        matrix (list): A list of lists representing a matrix
    Returns:
        bool: True if the matrix is CR-hamiltonian, False otherwise
    """
    for i in range(len(matrix) - 1):
        graph = defaultdict(list)
        for j in range(len(matrix)):
            graph[matrix[i][j]].append(matrix[i + 1][j])
            graph[matrix[i + 1][j]].append(matrix[i][j])
        if not is_hamiltonian_cycle(graph):
            print(f"=============\nMatrix is not CR-hamiltonian, Problem row: {i}")
            return False
    return True


def list_all_symbol_pairs(length):
    """
    generates all pairs (not just consecutive) of symbols
    Parameters:
        length (int): The length of the matrix
    Returns:
        list: A list of tuples containing all possible pairs of symbols
    """
    return combinations(range(length), 2)


def symbol_hamiltonian_checker(matrix):
    """
    checks if a matrix is symbol-hamiltonian. i.e. if all pairs of symbols form a hamiltonian cycle
    Parameters:
        matrix (list): A list of lists representing a matrix
    Returns:
        bool: True if the matrix is symbol-hamiltonian, False otherwise
    """
    pairs = list_all_symbol_pairs(len(matrix))
    for pair in pairs:
        graph = map_symbol_pairs_to_graph(pair, matrix)

        if not is_hamiltonian_cycle(graph):
            print(
                f"=============\nMatrix is not symbol-hamiltonian, Problem pair: {pair}"
            )
            return False
    return True


def anti_row_hamiltonian_checker(matrix):
    """
    Checks if a given Latin Square is anti-row-hamiltonian, i.e., all pairs of rows do not form a hamiltonian cycle
    Parameters:
        matrix (list): A list of lists representing a matrix
    Returns:
        bool: True if the matrix is anti-row-hamiltonian, False otherwise
    """
    pairs = list_all_symbol_pairs(len(matrix))
    for row1, row2 in pairs:
        graph = defaultdict(list)
        for j in range(len(matrix)):
            graph[matrix[row1][j]].append(matrix[row2][j])
            graph[matrix[row2][j]].append(matrix[row1][j])
        if is_hamiltonian_cycle(graph):
            return False
    return True
            

################################################################### Isotopy Tools #########################################################################

def apply_isotopy(square, row_perm, col_perm, sym_perm):
    """
    Applies an isotopy to the given square ->
    new_square[i][j] = sym_perm[square[row_perm[i]][col_perm[j]]]
    Parameters:
        square (list): A list of lists representing a square
        row_perm (list): A permutation of the rows
        col_perm (list): A permutation of the columns
        sym_perm (list): A permutation of the symbols
    Returns:
        list: A new square with the isotopy applied
    """
    n = len(square)
    new_square = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            new_square[i][j] = sym_perm[square[row_perm[i]][col_perm[j]]]
    return new_square


###### WARNING ########## THIS IS A STUPID AND SLOW WAY OF ISOTOPY CHECKING, DO NOT USE THIS FOR LARGE MATRICES
def are_isotopic(square1, square2):
    """
    Checks if square1 and square2 are isotopic

    DO NOT USE THIS FOR LARGE MATRICES
    Parameters:
        square1 (list): A list of lists representing a square
        square2 (list): A list of lists representing a square
    Returns:
        bool: True if there exist permutations of rows, columns, and symbols that transform square1 into square2
    """
    if len(square1) != len(square2):
        return False  # different sizes

    n = len(square1)

    # Brute force over all triples of permutations.
    for row_perm in permutations(range(n)):
        for col_perm in permutations(range(n)):
            for sym_perm in permutations(range(n)):
                if apply_isotopy(square1, row_perm, col_perm, sym_perm) == square2:
                    return True
    return False

################################################################ Generation Tools #################################################

def generate_latin_square(n):
    """
    Generates a latin square using backtracking and randomness
    Parameters:
        n (int): The order of the Latin square
    Returns:
        list: A list of lists representing a Latin square
    """

    def is_valid(board, row, col, num):
    # Check if num is already used in the current row or column.
        n = len(board)
        for i in range(n):
            if board[row][i] == num or board[i][col] == num:
                return False
        return True
    
    def backtrack(board, row, col):
        n = len(board)
        if row == n:
            # All rows filled; Latin square complete.
            return True

        # Compute next cell's position.
        next_row, next_col = (row, col + 1) if col < n - 1 else (row + 1, 0)
        
        # Try numbers in a random order for variety.
        numbers = list(range(1, n + 1))
        random.shuffle(numbers)
        
        for num in numbers:
            if is_valid(board, row, col, num):
                board[row][col] = num
                if backtrack(board, next_row, next_col):
                    return True
                board[row][col] = 0  # Backtrack: reset cell.
        return False
    board = [[0] * n for _ in range(n)]
    if backtrack(board, 0, 0):
        return board
    else:
        return None

def Z_n_LS_generator(n):
    """
    Generates the Z_n Latin Square of order n.
    Parameters:
        n (int): order of the Latin Square
    Returns:
        list: a Latin Square of order n
    """
    return [[(i+j)%n for j in range(n)] for i in range(n)]


def generate_latin_square_from_strong_starter(strong_starter):
    """
    Generates a Latin square from a strong starter
    Parameters:
        strong_starter (list): A list of pairs representing a strong starter
    Returns:
        list: A list of lists representing a Latin square
    """
    def find_pair(key, strong_starter):
        for pair in strong_starter:
            if pair[0] == key:
                return pair[1]
            if pair[1] == key:
                return pair[0]
        
    n = len(strong_starter) * 2 + 1
    latin_square = [[0] * n for _ in range(n)]
    latin_square[0][0] = 0
    for i in range(1, n):
        latin_square[0][i] = find_pair(i, strong_starter)

    #from row i to row i+1, add 1 to each element and shift over one spot right
    for row in range(1, n):
        for col in range(n):
            latin_square[row][col] = (latin_square[row-1][col] + 1) % n
        latin_square[row] = [latin_square[row].pop()] + latin_square[row]
    
    return latin_square


################################################################### Other Tools ###########################################################################

def valid_latin_square(matrix):
    """
    Checks if a given matrix forms a valid latin square

    Parameters:
        matrix (list): A list of lists representing a matrix
    Returns:
        bool: True if the matrix is a valid Latin square, False otherwise
    """
    for row in matrix:
        if len(set(row)) != len(row):
            return False
    for col in range(len(matrix)):
        if len(set(matrix[row][col] for row in range(len(matrix)))) != len(matrix):
            return False
    return True

def has_two_cycles(row1, row2):
    """
    Checks if the given two rows form only 2-cycles
    Parameters:
        row1 (list): A list of elements
        row2 (list): A list of elements
    Returns:
        bool: True if the two rows form only 2-cycles, False otherwise
    """
    # Create a mapping from elements in row1 to row2.
    mapping = {a: b for a, b in zip(row1, row2)}
    for a in mapping:
        # Check for a fixed point
        if mapping[a] == a:
            return False
        # Check that applying the mapping twice returns to the original element.
        if mapping.get(mapping[a]) != a:
            return False
    return True

def two_cycles_only(latin_square):
    """
    Checks if a Latin square has the property: every pair of rows forms only 2-cycles
    Parameters:
        latin_square (list): A list of lists representing a Latin square
    Returns:
        bool: True if the Latin square has the property: every pair of rows forms only 2-cycles, False otherwise
    """
    n = len(latin_square)
    # Check every distinct pair of rows.
    for i in range(n):
        for j in range(i + 1, n):
            if not has_two_cycles(latin_square[i], latin_square[j]):
                return False
    return True


def are_orthogonal(latin_square1, latin_square2):
    """
    Checks if two Latin squares are orthogonal
    Parameters:
        latin_square1 (list): A list of lists representing a Latin square
        latin_square2 (list): A list of lists representing a Latin square
    Returns:
        bool: True if the two Latin squares are orthogonal, False otherwise
    """
    n = len(latin_square1)
    if n != len(latin_square2):
        return False
    required_pairs = set()
    for i in range(n):
        for j in range(n):
            required_pairs.add((i, j))
    for i in range(n):
        for j in range(n):
            if (latin_square1[i][j], latin_square2[i][j]) in required_pairs:
                required_pairs.remove((latin_square1[i][j], latin_square2[i][j]))
            else:
                return False
    return True


def is_idempotent(latin_square):
    """
    Checks if a Latin square is idempotent, i.e., the main diagonal contains each symbol exactly once
    Parameters:
        latin_square (list): A list of lists representing a Latin square
    Returns:
        bool: True if the Latin square is idempotent, False otherwise
    """
    n = len(latin_square)
    return set(latin_square[i][i] for i in range(n)) == set(range(n))


def is_magic_square(matrix):
    """
    Checks if a given matrix is a magic square, i.e., all rows, columns, and diagonals have the same sum
    Parameters:
        matrix (list): A list of lists representing a matrix
    Returns:
        bool: True if the matrix is a magic square, False
    """
    n = len(matrix)
    # Ensure the matrix is square
    if any(len(row) != n for row in matrix):
        return False

    # Calculate the magic constant using the first row
    magic_sum = sum(matrix[0])

    # Check sums of all rows
    for row in matrix:
        if sum(row) != magic_sum:
            print(f"Row sum: {sum(row)}")
            print(f"Magic sum: {magic_sum}")
            return False

    # Check sums of all columns
    for col in range(n):
        if sum(matrix[row][col] for row in range(n)) != magic_sum:
            print(f"Column sum: {sum(matrix[row][col] for row in range(n))}")
            print(f"Magic sum: {magic_sum}")
            return False

    # Check the primary diagonal
    if sum(matrix[i][i] for i in range(n)) != magic_sum:
        print(f"Primary diagonal sum: {sum(matrix[i][i] for i in range(n))}")
        print(f"Magic sum: {magic_sum}")
        return False

    # Check the secondary diagonal
    if sum(matrix[i][n - 1 - i] for i in range(n)) != magic_sum:
        print(f"Secondary diagonal sum: {sum(matrix[i][n - 1 - i] for i in range(n))}")
        print(f"Magic sum: {magic_sum}")
        return False

    return True


def make_magic_square(matrix1, matrix2):
    """
    Makes a magic square from two orthogonal Latin squares with idemopotent and non-main diagonal idempotent properties
    Parameters:
        matrix1 (list): A list of lists representing a Latin square
        matrix2 (list): A list of lists representing a Latin square
    Returns:
        list: A list of lists representing a magic square if it exists, none otherwise
    """
    def numberToBase(n, b):
        #converts number n to base b
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
    
    if not are_orthogonal(matrix1, matrix2):
        #can't generate from non-orthogonal latin squares
        return None
    
    #procedure:
    # For each i, j in the matrix, we take the number formed by concatenating
    # the i, jth elements of the two matrices and convert it to base n (order of LS)
    magic_square = []
    base = len(matrix1)
    for i in range(base):
        row = []
        for j in range(base):
            new_num = map(str, numberToBase(int(str(matrix1[i][j])+str(matrix2[i][j])), base))
            num = int("".join(new_num))
            row.append(num)
        magic_square.append(row)
    return magic_square if is_magic_square(magic_square) else None


def print_latin_square(board):
    """
    Prints a given board with width adjusted
    Parameters:
        board (list): A list of lists representing a board
    """
    n = len(board)
    # Determine the width based on the largest number
    width = len(str(n))
    for row in board:
        print(" ".join(f"{num:>{width}}" for num in row))


def pretty_print_matrix(matrix):
    """
    Pretty prints an arbitrary matrix based on the maximum length of an element
    Parameters:
        matrix (list): A list of lists representing a matrix
    """
    max_length = max(len(str(cell)) for row in matrix for cell in row)
    for row in matrix:
        print(" ".join(f"{cell:^{max_length+2}}" for cell in row))

