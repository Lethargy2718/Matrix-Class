#i exit a lot while debugging
from sys import exit
import copy
from collections import defaultdict

#just a line i like to print to separate stuff
def l():
    print('--------------------------------------')
            
class Matrix:
    def __init__(self, data) -> None:
        #self explanatory stuff. well, to start, my matrices are lists of smaller lists. but i also made subcases for lists made of
        #integers and floats to have row and column vectors. you will see that diving deeper into the code. 
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
        #checks the validity of a matrix. also creates an identity matrix if you input an integer when initializing the matrix
        data = self.data
        if not isinstance(data, list):
            if isinstance(data, int):
                self.identity()
                return
            else:    
                raise ValueError("Matrix must be a list.")
        
        for i in range(0,len(data)):
            if not isinstance(data[i], (list, int, float)):
                raise ValueError("Rows must be lists or numbers.") 
            if isinstance(data[i], list):
                if len(data[i]) != len(data[0]):
                    raise ValueError("Rows must be equal in length.")
            
    def find_order(self):
        #finds the order of a matrix.
        if isinstance(self.data[0], list): 
            m = len(self.data)
            n = len(self.data[0])
        else:
            m = 1
            n = len(self.data)
        return (m,n)
         
    def check_vector(self):
        #checks whether the matrix is a row vector, column vector, or not a vector at all.
        if not isinstance(self.data[0], list):
            return 1 #row vector
        elif len(self.data[0]) == 1:
            return 2 #column vector
        else:
            return 0 #not a vector
        
    def dot_product(self, vec1, vec2):
        #dot product of two vectors
        if not isinstance(vec1, list) or not isinstance(vec2, list):
            raise ValueError('Vectors must be lists.')
        if len(vec1) != len(vec2):
            raise ValueError('Vectors must be equal in length.')
        
        product = 0
        for i in range(len(vec1)):
            product += vec1[i] * vec2[i]
            
        return product
            
    def get_column(self, index):
        #makes a list out of a column using its index.
        column = []
        columns = self.find_order()[1]
        if index >= columns:
            raise ValueError("Column index doesn't exist")
        
        for i in range(len(self.data)):
            column.append(self.data[i][index])
        return column 

    def check_zeroes(self, index):
        #checks if a row is only zeroes. useless but eh i am keeping it because why not
        if index >= self.m:
            raise ValueError("Index out of range.")
        
        for i in range(self.n):
            if self.data[index][i] != 0:
                return False
        return True    

    def sum_rows(self, index1, index2, multiple):
        #multiplies a row by a number then adds i to another
        if not isinstance(multiple, (int, float)):
            raise ValueError("Not a valid multiple.")
        if index1 >= self.m or index2 >= self.m:
            raise ValueError("Index out of range.")
        row3 = []
        for i in range(self.order[1]):
            row3.append(multiple * self.data[index1][i] + self.data[index2][i])
            
        return row3    
            
    def switch_rows(self, index1, index2):
        #switches two rows in a matrix if they exist
        if index1 >= self.m or index2 >= self.m:
            raise ValueError("Index out of range.")
        self.data[index1], self.data[index2] = self.data[index2], self.data[index1]
              
    def put_zeroes_down(self):
        #places the zero rows in a matrix at its bottom. kinda useless. i thought i would need it for rref
        #but i didn't. i am keeping it because it was fun to create
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
        #multiplies a vector by a scalar
        if isinstance(vec, (list)):
            for i in range(len(vec)):
                vec[i] *= multiple
        elif isinstance(vec, (float, int)):
            vec *= multiple
        else:
            raise ValueError("Not a valid vector.")     
         
    def identity(self):
        #creates an identity matrix of arbitary size
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
        return Matrix(self.data)
    
    def rank(self):
        #the rank of the matrix aka number of pivots/pivot columns
        return len(self.rref(True)[1])
    
    def pivot_columns(self):
        #returns the indices of pivot columns
        return self.rref(True)[1]
    
    def free_columns(self, pivotcols=None):
        #returns the indices of free columns
        if not pivotcols: pivotcols = self.pivot_columns()
        freecols = []
        for i in range(self.n):
            if i not in pivotcols:
                freecols.append(i)
        return freecols        
    
    def analyze(self):
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
        # multiplies two matrices
        rows1 = self.find_order()[0]
        cols1 = self.find_order()[1]
        
        rows2 = matrix.find_order()[0]
        cols2 = matrix.find_order()[1]
        
        if cols1 != rows2:
            raise ValueError('Invalid. columns of matrix 1 should be equal in number to Rows of matrix 2')

        # new matrix will be rows1 x cols2, so rows1 rows and cols2 columms
        new_matrix = []
        for i in range(rows1):
            new_matrix.append([])
            for j in range(cols2):
                aij = self.dot_product(self.data[i], matrix.get_column(j))
                new_matrix[i].append(aij)
                
        return Matrix(new_matrix)        
                
    def transpose(self):
        #transposes a matrix
        new_matrix = []
        for i in range(self.order[1]):
            new_matrix.append(self.get_column(i))
            
        return Matrix(new_matrix)    
    
    def _add(self, other):
        if self.order != other.order:
            raise ValueError('Matrices must be the same size.')
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])

    def __add__(self, other):
        return self._add(other)

    def __sub__(self, other):
        # multiplies the other matrix by -1 then adds which is the equivalent for subtraction
        return self._add(Matrix([[-value for value in row] for row in other.data]))
    
    def scalar_product(self, scalar):
        #multiplies a matrix by a scalar
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
        # finds the row reduced echelon form of a matrix
        
        # next_pivot_row is how i create the "echelon" form. it's how i keep track of the pivots locations and only place them
        # in a staircase manner
        new_matrix = self.copy()
        
        if augmented_vector:
            if not isinstance(augmented_vector, list):
                raise ValueError("Solution vector must be a list.")
           
            if len(augmented_vector) != new_matrix.m:
                raise ValueError("Solution vector is of invalid length. Must be equal to the number of rows.")
             
        pivot_columns = []
        next_pivot_row = 0
        
        for j in range(new_matrix.n): 
            # finding the first non zero entry in a column that's also in or below the next_pivot_row. can't be above it; that
            # wouldn't be an echelon at all.
            index, pivot = next(((x, y) for x,y in enumerate(new_matrix.get_column(j)) if y != 0 and x >= next_pivot_row), (None, None))
            if pivot:
                pivot_columns.append(j)
                #if the next pivot isn't in the right place, i switch rows to place it there.
                if index != next_pivot_row:
                    new_matrix.switch_rows(index, next_pivot_row)
                    if augmented_vector: 
                        augmented_vector[index], augmented_vector[next_pivot_row] = augmented_vector[next_pivot_row], augmented_vector[index]
                    
                # normalizing the pivot row    
                new_matrix.scalar_vector_product(new_matrix.data[next_pivot_row], 1/pivot)
                if augmented_vector: augmented_vector[next_pivot_row] *= 1/pivot

                column_mapping = enumerate(new_matrix.get_column(j))
                # iterating over every row and making it equal to zero using the pivot as long as it's not the pivot row itself
                # and not equal to 0 at the first place
                for row, num in column_mapping:
                    if row == next_pivot_row or num == 0:
                        continue
                    new_matrix.data[row] = new_matrix.sum_rows(next_pivot_row, row, -num)
                    if augmented_vector: augmented_vector[row] -= num * augmented_vector[next_pivot_row]
                # setting the next step of the echelon    
                next_pivot_row += 1  
                        
        #and that's it :)
        
        results = [new_matrix]
        #returns [matrix, pivot columns indices, augmented vector] if they exist
        if return_pivots: results.append(pivot_columns)
        else: results.append(None)    
        if augmented_vector: results.append(augmented_vector)
        else: results.append(None)
    
        return results
        
    def augment_vector(self, augmented_vector):
        if not isinstance(augmented_vector, list):
            raise ValueError("Vector must be a list.")
            
        new_matrix = self.copy()
        if len(augmented_vector) != new_matrix.m:
            raise ValueError("Solution vector is of invalid length. Must be equal to the number of rows.")
        for i in range(self.m):
            new_matrix.data[i].append(augmented_vector[i])    
        return new_matrix        

    def nullspace(self, results=None):
        # initializing data
        if not results: 
            results = self.rref(True)
        rref, pivot_cols = results[0], results[1]
        free_cols = self.free_columns(pivot_cols)
        rank = len(pivot_cols)
        equations, bases = [], []

        
        #TODO fix any nuances arising from this
        # if independent, return an empty list    
        if (self.m == self.n == rank) or (self.m > self.m and rank == self.n):
            return []
        
        # constructing a list of equations
        for i, row in enumerate(rref.data):
            equation = []
            for index, entry in enumerate(row):
                # skip zero coefficients and pivot variables. we write in terms of free vars
                if entry == 0 or index in pivot_cols:
                    continue   
                # each list is an equation, each tuple is a 'variable' of index tuple[0], coefficient tuple[1]  
                equation.append((index,-entry))
                
            # empty equations that arise when the row's free vars = 0 are skipped 
            if equation:
                equations.append(equation)
                          
        # constructing basis vectors    
        for i in range(len(free_cols)):
            new_basis = [0] * self.n
            for index, equation in enumerate(equations):
                p = pivot_cols[index]
                # sets the value to (coefficient * free variable current value)
                for var_index, coefficient in equation:
                    # 2nd part returns 1 if equal 0 otherwise which is equivalent to setting one free variable to 1 and the
                    # others to zero for each i
                    if len(free_cols) > 1:
                        value = coefficient * (free_cols[i] == var_index) 
                    else:
                        value = coefficient    
                    new_basis[p] += value
            # sets the free variable row to 1. the other free var rows are zero by default anyways
            if len(free_cols) > 1:
                new_basis[free_cols[i]] += 1 
            else:
                new_basis[free_cols[0]] += 1
                                                
            bases.append(new_basis)  
        return bases
                
    def solve_equations(self, sol):
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
            to solve a system with infinite solutions Ax = b:
            x = xp + N(A)
            where xp is any solution and N(A) is the nullspace
            i get both and format them into an answer.
            """
            print('Infinite solutions. General solution:')
            l()
            
            # finding the particular solution
            xp = [0] * self.n
            for i, row in enumerate(R.data):
                if i > rank - 1:
                    break
                xp[pivot_cols[i]] += b[i] / row[pivot_cols[i]]
                
            # obtaining the nullspace and appending xp to it    
            solution = self.nullspace()
            solution.append(xp)
            
            answers = defaultdict(list)
            
            # mapping each variable to its corresponding expression
            # basis vecs are stored as tuples with letters representing parameters and the paramters' coefficients
            for i, vector in enumerate(solution):    
                letter = chr(97 + i)
                for j, n in enumerate(vector):
                    # skip coefficients of value 0
                    if n != 0:
                        # the final item of 'solution' is xp. no letters/paramters added to that
                        if i == len(solution) - 1:
                            answers[j].append(n)
                            continue
                        answers[j].append((letter, n))
            
            # formatting the dictionary into text
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
                        
                    # removing the extra plus sign    
                if text[-1] == ' ':
                    text = text[:-2]    
                print(f'x{index + 1}: ' + text)
                
        if type == 2:
            print('No solution.')
          
# my testing playground. just try whatever here   

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


# unique solution example
matrix = Matrix([[7,5], [3,-4]])
b = [-12, 1]
matrix.solve_equations(b)

l()
l()

# no solution example
matrix = Matrix([[1,2], [2,4]])
b = [4,9]
matrix.solve_equations(b)

