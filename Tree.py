








import random

class TreeContainer:

    def __init__(self):
        # ultimate parent isn't counted as a real node. It is only here so the root node also has a parent, which makes implementation easier.
        self.ultimate_parent = TreeNode(False, None, buffer, 1)
        self.list_of_nodes = list()
    

    # The user of this function should run this on two different trees, therefore getting two nodes: node1 and node2.
    # Then they should find the parent node for each of these nodes: parent1 and parent2 respectively.
    # They should compare pointers in each of the parents to find out what child to exchange:
    # e.g. parent1 has two children. We need to find out which of these two children is node1. Namely, this child will then be replaced.
    # Then the user should simply set node2 to the found location, and do the same process for parent2.
    # This will mean an exchange of entire subtrees.

    # Then the user should do a walk over the entire tree to update the list of nodes in the tree container.

    
    # Returns pointer to random node in the tree.
    def split_for_crossover(self):
        num_of_nodes = len(self.list_of_nodes)
        # the interval is incluseive
        exchanged_node = self.list_of_nodes(random.randint(0,num_of_nodes-1))
    

    # def mutation







class TreeNode:
    def __init__(self, is_leaf: bool, numeric_vector=None, elem_func=None, elem_func_num_of_args=0):
        self.is_leaf = is_leaf
        self.numeric_vector = numeric_vector

        self.elem_func = elem_func
        self.elem_func_num_of_args = elem_func_num_of_args

        self.parent_node = None
        self.child_one = None
        self.child_two = None
    
    def calculate(self):
        if(self.is_leaf):
            return self.numeric_vector
        elif(self.elem_func_num_of_args == 2):
            return self.elem_func(self.child_one.calculate(), self.child_two.calculate()) 
        else:
            return self.elem_func(self.child_one.calculate())
        



# Elementary functions

# These take one or two numpy arrays and apply a function on them elementwise to create a new array.

def buffer(first_array):
    return first_array

def addition(first_array, second_array):
    return_array = first_array + second_array
    return return_array

def subtraction(first_array, second_array):
    return_array = first_array - second_array
    return return_array

def multiplication(first_array, second_array):
    return_array = first_array * second_array
    return return_array

def division(first_array, second_array):
    return_array = first_array / second_array
    return return_array


def exponentiation(first_array, second_array):
    return_array = first_array ** second_array
    return return_array