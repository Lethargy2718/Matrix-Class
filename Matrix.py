from sys import exit # For debugging purposes
import copy
from collections import defaultdict


def l():
    """
    To print a not-so-fancy line that separates
    things in the terminal.
    """
    print('--------------------------------------')
            
class Matrix:
    def __init__(self, data) -> None:
        """
        Matrices are nested lists. A row vector subcase also exists. 
        """
        self.data = copy.deepcopy(data)
        self.check_validity()
        self.order = self.find_order()
        self.m = self.order[0]
        self.n = self.order[1]
        self.vector = self.check_vector()
        
    def __str__(self) -> str:
        matrix = ''
        for i in range(len(self.data)):
            matrix += str(self.data[i])
            if i < len(self.data) - 1:
                matrix += '\n'    
        return matrix 
            
    def __repr__(self) -> str:
        matrix = ''
        for i in range(len(self.data)):
            matrix += str(self.data[i])
            if i < len(self.data) - 1:
                matrix += '\n'    
        return matrix   

# Helper functions:
    def check_validity(self):
        """
        checks the validity of a matrix. Also creates an identity matrix
        if you input an integer when initializing the matrix.
        """
        data = self.data
        if not isinstance(data, list):
            if isinstance(data, int):
                self.identity()
                return
            else:    
                raise ValueError("Matrix must be a list.")
        
        for i in range(0,len(data)):
            if not isinstance(data[i], (list, int, float)):
                raise ValueError("Rows must be lists, integers, or floats.") 
            if isinstance(data[i], list):
                if len(data[i]) != len(data[0]):
                    raise ValueError("Rows must be equal in length.")
            
    def find_order(self):
        """
        Finds the order of a matrix.
        """
        if isinstance(self.data[0], list): 
            m = len(self.data)
            n = len(self.data[0])
        else:
            m = 1
            n = len(self.data)
        return (m,n)
         
    def check_vector(self):
        """
        Checks whether the matrix is a row vector, 
        column vector, or not a vector at all.
        """
        if not isinstance(self.data[0], list):
            return 1 #row vector
        elif len(self.data[0]) == 1:
            return 2 #column vector
        else:
            return 0 #not a vector
        
    def dot_product(self, vec1, vec2):
        """
        Calculates the dot product of two vectors.
        """
        if not isinstance(vec1, list) or not isinstance(vec2, list):
            raise ValueError('Vectors must be lists.')
        if len(vec1) != len(vec2):
            raise ValueError('Vectors must be equal in length.')
        
        product = 0
        for i in range(len(vec1)):
            product += vec1[i] * vec2[i]
            
        return product
            
    def get_column(self, index):
        """
        Returns a list out of a column using its index.
        """
        column = []
        columns = self.find_order()[1]
        if index >= columns:
            raise ValueError("Column index doesn't exist")
        
        for i in range(len(self.data)):
            column.append(self.data[i][index])
        return column 

    def check_zeroes(self, index):
        """
        Checks if a row is a zero vector.
        """
        if index >= self.m
            raise ValueError("Index out of range.")
        
        for i in range(self.n):
            if self.data[index][i] != 0:
                return False
        return True    

    def sum_rows(self, index1, index2, multiple):
        """
        Multiplies a row by a scalar then adds it to another row.
        The other row's value is updated to be the new value.
        """
        
        if not isinstance(multiple, (int, float)):
            raise ValueError("Not a valid multiple.")
        if index1 >= self.m or index2 >= self.m:
            raise ValueError("Index out of range.")
        row3 = []
        for i in range(self.order[1]):
            row3.append(multiple * self.data[index1][i] + self.data[index2][i])
            
        return row3    
            
    def switch_rows(self, index1, index2):
        """
        Switches two rows.
        """
        if index1 >= self.m or index2 >= self.m:
            raise ValueError("Index out of range.")
        self.data[index1], self.data[index2] = self.data[index2], self.data[index1]
              
    def put_zeroes_down(self):
        """
        Moves all the zero rows to the bottom of the matrix.
        """
        m = self.m
        zeroes = []
        switched = False
        for i in range(m):
            if i not in zeroes:
                zero_row = self.check_zeroes(i)
                if zero_row:
                    switched = False
                    for j in range(m-1, i, -1):
                        if not self.check_zeroes(j):
                            self.switch_rows(i,j)
                            switched = True
                            zeroes.append(j)
                            break
                        elif j not in zeroes: 
                            zeroes.append(j)
                    if not switched:
                        zeroes.append(i)         
        return len(zeroes)                    
                   
    def scalar_vector_product(self, vec, multiple):
        """
        Multiplies a vector by a scalar.
        """
        if isinstance(vec, (list)):
            for i in range(len(vec)):
                vec[i] *= multiple
        elif isinstance(vec, (float, int)):
            vec *= multiple
        else:
            raise ValueError("Not a valid vector.")     
         
    def identity(self):
        """
        Creates an identity matrix of arbitary size.
        """
        matrix = []
        size = self.data
        if size == 0:
            raise ValueError("Identity matrices cannot be of size 0.")
        for i in range(size):
            matrix.append([])
            for j in range(size):
                if i==j:
                    matrix[i].append(1)
                else:
                    matrix[i].append(0)   
        self.data = matrix                       

    def copy(self):
        """
        Returns a copy of the matrix.
        """
        return Matrix(self.data)
    
    def rank(self):
        """
        Returns the rank of the matrix.
        """
        return len(self.rref(True)[1])
    
    def pivot_columns(self):
        """
        Returns a list of the indices of pivot columns.
        """
        return self.rref(True)[1]
    
    def free_columns(self, pivotcols=None):
        """
        Returns a list of the indices of free columns.
        """
        if not pivotcols: pivotcols = self.pivot_columns()
        freecols = []
        for i in range(self.n):
            if i not in pivotcols:
                freecols.append(i)
        return freecols        
    
    def analyze(self):
        """
        Prints general info on the matrix.
        """
        results = self.rref(True)
        l()
        print("Matrix: ")
        self.print()
        l()
        print("Rref: ")
        print(results[0])
        l()
        print('Order: ', self.order)
        print('Rank: ', len(results[1]))
        print('Pivot columns: ', self.pivot_columns())
        print('Free columns: ', self.free_columns())
        print('Nullspace basis: ', self.nullspace())
        l()
        
# Matrix operations:
    def matrix_product(self, matrix):
        """
        Returns the product of two matrices.
        """
        rows1 = self.find_order()[0]
        cols1 = self.find_order()[1]
        
        rows2 = matrix.find_order()[0]
        cols2 = matrix.find_order()[1]
        
        if cols1 != rows2:
            raise ValueError('Invalid. columns of matrix 1 should be equal in number to Rows of matrix 2')

        # New matrix will be rows1 x cols2, so rows1 rows and cols2 columms.
        new_matrix = []
        for i in range(rows1):
            new_matrix.append([])
            for j in range(cols2):
                aij = self.dot_product(self.data[i], matrix.get_column(j))
                new_matrix[i].append(aij)
                
        return Matrix(new_matrix)        
                
    def transpose(self):
        """
        Returns the transpose of the matrix.
        """
        new_matrix = []
        for i in range(self.order[1]):
            new_matrix.append(self.get_column(i))
            
        return Matrix(new_matrix)    
    
    def _add(self, other):
        """
        Returns the sum of two matrices.
        """
        if self.order != other.order:
            raise ValueError('Matrices must be the same size.')
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])

    def __add__(self, other):
        """
        Calls the private method _add to add then returns the final matrix.
        """
        return self._add(other)

    def __sub__(self, other):
        """
        Calls the private _add method to subtract after self multiplication by -1
        then returns the final matrix.
        """
        return self._add(Matrix([[-value for value in row] for row in other.data]))
    
    def scalar_product(self, scalar):
        """
        Returns the product of the matrix with a scalar.
        """
        data = self.data
        if not isinstance(scalar, (int, float)):
            raise ValueError("Not a valid scalar.")
        if self.vector == 1:
            for i in range(self.order[1]):
                data[i] *= scalar      
        else:
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    data[i][j] *= scalar   
        
        return Matrix(data)            
        
    def rref(self, return_pivots=False, augmented_vector=None):
        """
        Returns the row reduced echelon form of a matrix.
        """
        new_matrix = self.copy()
        
        if augmented_vector:
            if not isinstance(augmented_vector, list):
                raise ValueError("Solution vector must be a list.")
           
            if len(augmented_vector) != new_matrix.m:
                raise ValueError("Solution vector is of invalid length. Must be equal to the number of rows.")
                
        # next_pivot_row is how I create the "echelon" form. It's how I keep
        # track of the pivots locations and only place them in a staircase manner.   
        pivot_columns = []
        next_pivot_row = 0
        
        for j in range(new_matrix.n): 
            # Finding the first non zero entry in a column that's also in or below
            # the next_pivot_row. can't be above it; that wouldn't be an echelon at all.
            index, pivot = next(((x, y) for x,y in enumerate(new_matrix.get_column(j)) if y != 0 and x >= next_pivot_row), (None, None))
            if pivot:
                pivot_columns.append(j)
                # If the next pivot isn't in the right place, switch rows to place it there.
                if index != next_pivot_row:
                    new_matrix.switch_rows(index, next_pivot_row)
                    if augmented_vector: 
                        augmented_vector[index], augmented_vector[next_pivot_row] = augmented_vector[next_pivot_row], augmented_vector[index]
                    
                # Normalizing the pivot row    
                new_matrix.scalar_vector_product(new_matrix.data[next_pivot_row], 1/pivot)
                if augmented_vector: augmented_vector[next_pivot_row] *= 1/pivot

                column_mapping = enumerate(new_matrix.get_column(j))
                # Iterating over every row and making it equal to zero using the pivot as long as it's not the pivot row itself
                # and not equal to 0 at the first place.
                for row, num in column_mapping:
                    if row == next_pivot_row or num == 0:
                        continue
                    new_matrix.data[row] = new_matrix.sum_rows(next_pivot_row, row, -num)
                    if augmented_vector: augmented_vector[row] -= num * augmented_vector[next_pivot_row]
                # Setting the next step of the echelon    
                next_pivot_row += 1  
                                
        results = [new_matrix]
        # Returns [matrix, pivot columns indices, augmented vector]
        # Whatever doesn'r exist is None.
        if return_pivots: results.append(pivot_columns)
        else: results.append(None)    
        if augmented_vector: results.append(augmented_vector)
        else: results.append(None)
    
        return results
        
    def augment_vector(self, augmented_vector):
        """
        Returns the matrix augmented by a vector.
        """
        if not isinstance(augmented_vector, list):
            raise ValueError("Vector must be a list.")
            
        new_matrix = self.copy()
        if len(augmented_vector) != new_matrix.m:
            raise ValueError("Solution vector is of invalid length. Must be equal to the number of rows.")
        for i in range(self.m):
            new_matrix.data[i].append(augmented_vector[i])    
        return new_matrix        

    def nullspace(self, results=None):
        """
        Returns the basis of the nullspace/kernel.
        """
        # Initializing data
        if not results: 
            results = self.rref(True)
        rref, pivot_cols = results[0], results[1]
        free_cols = self.free_columns(pivot_cols)
        rank = len(pivot_cols)
        equations, bases = [], []

        #TODO 
        # Fix any nuances arising from this.
        # If independent, return an empty list    
        if (self.m == self.n == rank) or (self.m > self.m and rank == self.n):
            return []
        
        # Constructing a list of equations
        for i, row in enumerate(rref.data):
            equation = []
            for index, entry in enumerate(row):
                # Skip zero coefficients and pivot variables.
                # We write in terms of free variables.
                if entry == 0 or index in pivot_cols:
                    continue   
                # Each list is an equation, each tuple is
                # a 'variable' of index tuple[0], coefficient tuple[1].
                equation.append((index,-entry))
                
            # Empty equations that arise when the row's free variables == 0 are skipped.
            if equation:
                equations.append(equation)
                          
        # Constructing basis vectors    
        for i in range(len(free_cols)):
            new_basis = [0] * self.n
            for index, equation in enumerate(equations):
                p = pivot_cols[index]
                # Sets the value to (coefficient * free variable current value).
                for var_index, coefficient in equation:
                    # 2nd part returns 1 if equal 0 otherwise which is equivalent to setting
                    #  one free variable to 1 and the others to zero for each i
                    if len(free_cols) > 1:
                        value = coefficient * (free_cols[i] == var_index) 
                    else:
                        value = coefficient    
                    new_basis[p] += value
            # Sets the free variable row to 1.
            # The other free variable rows are zero by default anyways.
            if len(free_cols) > 1:
                new_basis[free_cols[i]] += 1 
            else:
                new_basis[free_cols[0]] += 1
                                                
            bases.append(new_basis)  
        return bases
                
    def solve_equations(self, sol):
        """
        Solves a system of equations.
        """
        type = None
        augmented_matrix = self.augment_vector(sol)
        results = self.rref(True, sol)
        results_aug = augmented_matrix.rref(True)
        R, pivot_cols, b = results[0], results[1], results[2]
        rank = len(results[1]) #A's rank
        rank_aug = len(results_aug[1]) #[A|B]'s rank
        
        for i in range(self.m):
            if self.check_zeroes(i) and b[i] != 0:
                print('System is not consistent.')
                return 
        if rank == rank_aug and rank == self.n:
            # unique solution
            type = 0
        if rank == rank_aug and rank < self.n:
            # infinite solutions
            type = 1    
        elif rank != rank_aug:
            # no solution
            type = 2
            
        if type == 0:
            print('Unique solution:')
            l()
            print(''.join(f'x{i+1} = {b[i]} ' for i in range(self.n)))
            
        if type == 1:    
            """
            To solve a system with infinite solutions Ax = b:
            x = xp + N(A)
            where xp is any solution and N(A) is the nullspace
            I get both and format them into an answer.
            """
            print('Infinite solutions. General solution:')
            l()
            
            # Finding the particular solution.
            xp = [0] * self.n
            for i, row in enumerate(R.data):
                if i > rank - 1:
                    break
                xp[pivot_cols[i]] += b[i] / row[pivot_cols[i]]
                
            # Obtaining the nullspace and appending xp to it. 
            solution = self.nullspace()
            solution.append(xp)
            
            answers = defaultdict(list)
            
            # Mapping each variable to its corresponding expression.
            # Basis vectors are stored as tuples with letters representing parameters
            # and the paramters' coefficients
            for i, vector in enumerate(solution):    
                letter = chr(97 + i)
                for j, n in enumerate(vector):
                    # Skip coefficients of value 0.
                    if n != 0:
                        # The final item of 'solution' is xp. 
                        # No letters/paramters added to that
                        if i == len(solution) - 1:
                            answers[j].append(n)
                            continue
                        answers[j].append((letter, n))
            
            # Formatting the dictionary into text
            for index, value in sorted(answers.items()):
                text = ''
                for object in value:
                    if isinstance(object, tuple):
                        if object[1] != 1:
                            text += str(object[1]) #num
                        text += object[0] #letter
                        text += ' + '
                    else:    
                        text += str(object)
                        
                # Removing the extra plus sign    
                if text[-1] == ' ':
                    text = text[:-2]    
                print(f'x{index + 1}: ' + text)
                
        if type == 2:
            print('No solution.')
          
# My testing playground:  

# infinite solutions examples
matrix = Matrix([[2,2,2], [2,3,2], [1,1,1]]) 
b = [-2,4,-1]
matrix.solve_equations(b)

l()
l()

matrix = Matrix([[1,1,1], [2,-2,-2], [3,-1,-1]])
b = [10,4,14]
matrix.solve_equations(b)

l()
l()

matrix = Matrix([[1,2], [2,4]])
b = [4,8]
matrix.solve_equations(b)

l()
l()


# Unique solution example
matrix = Matrix([[7,5], [3,-4]])
b = [-12, 1]
matrix.solve_equations(b)

l()
l()

# No solution example
matrix = Matrix[[2,4], [4,8])
b = [1,2]
matrix.solve_equations(b)
