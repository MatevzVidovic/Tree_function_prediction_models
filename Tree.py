








import random

import numpy as np






class TreeContainer:

    def __init__(self):
        # ultimate parent isn't counted as a real node. It is only here so the root node also has a parent, which makes implementation easier.
        self.ultimate_parent = TreeNode(False, None, pass_through, 1)
        self.set_of_nodes = set()
    

    def copy(self):
        new_self = TreeContainer()
        new_self.ultimate_parent.child_one = copy_subtree(self.ultimate_parent.child_one)
        new_self.ultimate_parent.child_one.parent_node = new_self.ultimate_parent
        add_nodes_from_subtree_to_set(new_self.ultimate_parent.child_one, new_self.set_of_nodes)

        return new_self

    
    


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
    
    def copy_without_parent_and_children(self):
        new_self = TreeNode(self.is_leaf, self.numeric_vector, self.elem_func, self.elem_func_num_of_args)
        return new_self
    
    def add_left_child(self, adding_node):
        self.child_one = adding_node
        adding_node.parent_node = self
        return
    
    def add_right_child(self, adding_node):
        self.child_two = adding_node
        adding_node.parent_node = self
        return


        


def add_nodes_from_subtree_to_set(subtree_root: TreeNode, goal_set: set):
        
        if subtree_root == None:
            return
        
        goal_set.add(subtree_root)

        add_nodes_from_subtree_to_set(subtree_root.child_one, goal_set)
        add_nodes_from_subtree_to_set(subtree_root.child_two, goal_set)

        return


# Watch out, because root has no parent. You need to assign it's parent to wherever you are copying to.
def copy_subtree(subtree_root: TreeNode):

    if subtree_root == None:
        return None

    new_current_node = subtree_root.copy_without_parent_and_children()

    if subtree_root.child_one != None:
        new_left_subtree = copy_subtree(subtree_root.child_one)
        new_current_node.child_one = new_left_subtree
        new_left_subtree.parent_node = new_current_node

    if subtree_root.child_two != None:
        new_right_subtree = copy_subtree(subtree_root.child_two)
        new_current_node.child_two = new_right_subtree
        new_right_subtree.parent_node = new_current_node
    
    return new_current_node
    

def crossover(coparent_1: TreeContainer, coparent_2: TreeContainer):

        # Since we want to keep the parent trees the same because we might want to use elitism in our algorithm, we will make new trees.
        parent_tree_1 = coparent_1.copy()
        parent_tree_2 = coparent_2.copy()


        # selects a random element of the set
        # You can't use random.choice on sets anymore, so we convert it to a tuple first.
        exchanged_node_1 = random.choice(tuple(parent_tree_1.set_of_nodes))
        exchanged_node_2 = random.choice(tuple(parent_tree_2.set_of_nodes))

        



        # e.g. parent_node_1 has two children. We need to find out which of these two children is exchange_node_1.
        # This child will then be replaced.
        # exchange_node_2 will take that place. And vice versa will happen for parent_node_2 and exchange_node_1.

        parent_node_1 = exchanged_node_1.parent_node
        parent_node_2 = exchanged_node_2.parent_node


        exchanged_node_1.parent_node = parent_node_2

        # try:
        #     parent_node_2_takes_child_one = parent_node_2.child_one == exchanged_node_1
        # except:
        #     print()
        #     print()
        #     print(exchanged_node_2)
        #     print_subtree(parent_tree_2.ultimate_parent.child_one)
        #     # print(parent_node_2)

        parent_node_2_takes_child_one = parent_node_2.child_one == exchanged_node_1


        if (parent_node_2_takes_child_one):
            parent_node_2.child_one = exchanged_node_1
        else:
            parent_node_2.child_two = exchanged_node_1



        exchanged_node_2.parent_node = parent_node_1

        parent_node_1_takes_child_one = parent_node_1.child_one == exchanged_node_1
        
        if (parent_node_1_takes_child_one):
            parent_node_1.child_one = exchanged_node_2
        else:
            parent_node_1.child_two = exchanged_node_2
        



        parent_tree_1.set_of_nodes = set()
        parent_tree_2.set_of_nodes = set()
        add_nodes_from_subtree_to_set(parent_tree_1.ultimate_parent.child_one, parent_tree_1.set_of_nodes)
        add_nodes_from_subtree_to_set(parent_tree_2.ultimate_parent.child_one, parent_tree_2.set_of_nodes)


        return parent_tree_1, parent_tree_2

def print_subtree(subtree_root: TreeNode, level_tuple=()):

    if subtree_root == None:
        return

    if subtree_root.is_leaf:
        print(str(level_tuple) + ":")
        print(subtree_root.numeric_vector)
    else:
        print(str(level_tuple) + ":")
        print(subtree_root.elem_func)
    
    print(subtree_root.parent_node)
    
    level_tuple_left = (level_tuple, "L")
    print_subtree(subtree_root.child_one, level_tuple_left)
    level_tuple_right = (level_tuple, "R")
    print_subtree(subtree_root.child_two, level_tuple_right)

    return








# Elementary functions

# These take one or two numpy arrays and apply a function on them elementwise to create a new array.

def pass_through(first_array):
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







# Unit test

testing_tree = TreeContainer()
const_5 = 5 * np.ones(10)
const_4 = 4 * np.ones(10)
const_2 = 2 * np.ones(10)
testing_tree.ultimate_parent.add_left_child(TreeNode(False, None, addition, 2))
testing_tree.ultimate_parent.child_one.add_left_child(TreeNode(True, const_5))
testing_tree.ultimate_parent.child_one.add_right_child(TreeNode(False, None, division, 2))
testing_tree.ultimate_parent.child_one.child_two.add_left_child(TreeNode(True, const_4))
testing_tree.ultimate_parent.child_one.child_two.add_right_child(TreeNode(True, const_2))

print(testing_tree.ultimate_parent.child_one.calculate())
print(testing_tree.ultimate_parent.calculate())

print(testing_tree.ultimate_parent.child_one.child_two.calculate())

testing_set = set()
add_nodes_from_subtree_to_set(testing_tree.ultimate_parent.child_one.child_two, testing_set)
# should be 3
print(len(testing_set))
print(testing_set)


testing_set = set()
add_nodes_from_subtree_to_set(testing_tree.ultimate_parent.child_one.child_one, testing_set)
# should be 1
print(len(testing_set))
print(testing_set)


testing_set = set()
add_nodes_from_subtree_to_set(testing_tree.ultimate_parent.child_one, testing_set)
# should be 5
print(len(testing_set))
print(testing_set)

print_subtree(testing_tree.ultimate_parent.child_one)






testing_tree_2 = TreeContainer()
const_5 = 5 * np.ones(10)
const_4 = 4 * np.ones(10)
const_2 = 2 * np.ones(10)
testing_tree_2.ultimate_parent.add_left_child(TreeNode(False, None, addition, 2))
testing_tree_2.ultimate_parent.child_one.add_left_child(TreeNode(True, const_5))
testing_tree_2.ultimate_parent.child_one.add_right_child(TreeNode(False, None, division, 2))
testing_tree_2.ultimate_parent.child_one.child_two.add_left_child(TreeNode(True, const_4))
testing_tree_2.ultimate_parent.child_one.child_two.add_right_child(TreeNode(True, const_2))



print_subtree(testing_tree_2.ultimate_parent.child_one)



offspring_1, offspring_2 = crossover(testing_tree, testing_tree_2)

print("offspring_1")
print_subtree(offspring_1.ultimate_parent.child_one)
print("offspring_2")
print_subtree(offspring_2.ultimate_parent.child_one)