# Latin Square Tools
A Python module offering a suite of tools to work with Latin squares, including Hamiltonicity checks, isotopy operations, generation algorithms, and additional validation utilities.

## Features
- Hamiltonicity Tools:
  - Check if a graph is a Hamiltonian cycle.
  - Verify CE-hamiltonian, CR-hamiltonian, and symbol-hamiltonian Latin squares.
  - Assess anti-row-hamiltonicity.
- Isotopy Tools:
  - Apply isotopies to Latin squares.
  - Check isotopy between two Latin squares (brute-force approach).
- Generation Tools:
  - Generate a random Latin square using backtracking.
  - Create the $Z_n$ Latin square.
  - Create a Latin square from a strong starter.
- Other Tools:
  - Validate Latin squares.
  - Verify orthogonality and idempotency.
  - Construct magic squares from orthogonal Latin squares.
  - Utilities for pretty printing matrices.

## Installation
No external dependencies are required. Simply clone or download the module into your project:
```bash
git clone https://github.com/GanschowJosh/LatinSquareTools.git
```

## Usage
Import the module and call the functions as needed. For example:
```python
from latin_square_tools import generate_latin_square, is_magic_square, print_latin_square

# Generate a Latin square of order 5
square = generate_latin_square(5)
print("Generated Latin Square:")
print_latin_square(square)

# Check if the generated square is magic
if is_magic_square(square):
    print("The square is a magic square!")
else:
    print("The square is not a magic square.")
```

## Contributions/Suggestions
Feel free to fork and add your own tools or send some suggestions for any utilities or functionality you'd like to see!
