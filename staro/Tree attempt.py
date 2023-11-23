








import random

import numpy as np






class TreeContainer:

    def __init__(self):
        # ultimate parent isn't counted as a real node. It is only here so the root node also has a parent, which makes implementation easier.
        self.ultimate_parent = TreeNode(False, None, pass_through, 1)
        self.list_of_nodes = list()
    

    def copy(self):
        new_self = TreeContainer()
        new_self.ultimate_parent.children[0] = copy_subtree(self.ultimate_parent.children[0])
        new_self.ultimate_parent.children[0].parent = new_self.ultimate_parent
        append_nodes_from_subtree_to_list(new_self.ultimate_parent.children[0], new_self.list_of_nodes)

        return new_self

    
    


    # def mutation







class TreeNode:

    # Children is a list. It will have no elements if the node is a leaf, one element if it is a function,
    # two elements if it is an operation.

    # We have chosen a list because it allows us to take whatever number of arguments we want.
    # It also allows us to check the number of children with len().

    # Visually we can think of the first element as the left child and the second element as the right child.
    # This visualization is important for non-commutative operation like division.

    def __init__(self, is_leaf: bool, numeric_vector=None, elem_func=None, elem_func_num_of_args=0):
        self.is_leaf = is_leaf
        self.numeric_vector = numeric_vector

        self.elem_func = elem_func
        self.elem_func_num_of_args = elem_func_num_of_args

        self.parent = None
        self.children = list()
        
    
    def calculate(self):
        if(self.is_leaf):
            return self.numeric_vector
        elif(self.elem_func_num_of_args == 1):
            return self.elem_func(self.children[0].calculate())
        elif(self.elem_func_num_of_args == 2):
            return self.elem_func(self.children[0].calculate(), self.children[1].calculate()) 
    

    def copy_without_parent_and_children(self):
        new_self = TreeNode(self.is_leaf, self.numeric_vector, self.elem_func, self.elem_func_num_of_args)
        return new_self
    
    def add_child(self, adding_node):
        self.children.append(adding_node)
        adding_node.parent = self
        return

    def reset_left_child(self, adding_node):
        try:
            self.children[0] = adding_node
            adding_node.parent = self
        except:
            error_msg = "Node doesn't have children yet. A child can't be reset before a place for them exists.\n"
            error_msg += "Use self.add_child(adding_node).\n"
            print(error_msg)
        return
    
    def reset_right_child(self, adding_node):
        try:
            self.children[1] = adding_node
            adding_node.parent = self
        except:
            error_msg = "Node doesn't have a second child yet. Current len is: " + str(len(self.children))
            error_msg += ". The second child can't be reset before a place for it exists.\n"
            error_msg += "Add the left child and then add the right child by using: self.add_child(adding_node).\n"
            error_msg += "There aren't assurances for this: If you absolutely have to add the right child right now, add None as the first child.\n"

            print(error_msg)
        return

        


def append_nodes_from_subtree_to_list(subtree_root: TreeNode, goal_list: list):
        
        if subtree_root == None:
            return
        
        goal_list.append(subtree_root)

        append_nodes_from_subtree_to_list(subtree_root.children[0], goal_list)
        append_nodes_from_subtree_to_list(subtree_root.children[1], goal_list)

        return


# Watch out, because root has no parent. You need to assign it's parent to wherever you are copying to.
def copy_subtree(subtree_root: TreeNode):

    if subtree_root == None:
        return None

    new_current_node = subtree_root.copy_without_parent_and_children()

    if subtree_root.children[0] != None:
        new_left_subtree = copy_subtree(subtree_root.children[0])
        new_current_node.children[0] = new_left_subtree
        new_left_subtree.parent = new_current_node

    if subtree_root.children[1] != None:
        new_right_subtree = copy_subtree(subtree_root.children[1])
        new_current_node.children[1] = new_right_subtree
        new_right_subtree.parent = new_current_node
    
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

        parent_node_1 = exchanged_node_1.parent
        parent_node_2 = exchanged_node_2.parent


        exchanged_node_1.parent = parent_node_2

        # try:
        #     parent_node_2_takes_children[0] = parent_node_2.children[0] == exchanged_node_1
        # except:
        #     print()
        #     print()
        #     print(exchanged_node_2)
        #     print_subtree(parent_tree_2.ultimate_parent.children[0])
        #     # print(parent_node_2)

        parent_node_2_takes_left_child = parent_node_2.children[0] == exchanged_node_1


        if (parent_node_2_takes_left_child):
            parent_node_2.children[0] = exchanged_node_1
        else:
            parent_node_2.children[1] = exchanged_node_1



        exchanged_node_2.parent = parent_node_1

        parent_node_1_takes_left_child = parent_node_1.children[0] == exchanged_node_1
        
        if (parent_node_1_takes_left_child):
            parent_node_1.children[0] = exchanged_node_2
        else:
            parent_node_1.children[1] = exchanged_node_2
        



        parent_tree_1.set_of_nodes = set()
        parent_tree_2.set_of_nodes = set()
        add_nodes_from_subtree_to_set(parent_tree_1.ultimate_parent.children[0], parent_tree_1.set_of_nodes)
        add_nodes_from_subtree_to_set(parent_tree_2.ultimate_parent.children[0], parent_tree_2.set_of_nodes)


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
    
    print(subtree_root.parent)
    
    level_tuple_left = (level_tuple, "L")
    print_subtree(subtree_root.children[0], level_tuple_left)
    level_tuple_right = (level_tuple, "R")
    print_subtree(subtree_root.children[1], level_tuple_right)

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
testing_tree.ultimate_parent.children[0].add_left_child(TreeNode(True, const_5))
testing_tree.ultimate_parent.children[0].add_right_child(TreeNode(False, None, division, 2))
testing_tree.ultimate_parent.children[0].children[1].add_left_child(TreeNode(True, const_4))
testing_tree.ultimate_parent.children[0].children[1].add_right_child(TreeNode(True, const_2))

print(testing_tree.ultimate_parent.children[0].calculate())
print(testing_tree.ultimate_parent.calculate())

print(testing_tree.ultimate_parent.children[0].children[1].calculate())

testing_set = set()
add_nodes_from_subtree_to_set(testing_tree.ultimate_parent.children[0].children[1], testing_set)
# should be 3
print(len(testing_set))
print(testing_set)


testing_set = set()
add_nodes_from_subtree_to_set(testing_tree.ultimate_parent.children[0].children[0], testing_set)
# should be 1
print(len(testing_set))
print(testing_set)


testing_set = set()
add_nodes_from_subtree_to_set(testing_tree.ultimate_parent.children[0], testing_set)
# should be 5
print(len(testing_set))
print(testing_set)

print_subtree(testing_tree.ultimate_parent.children[0])






testing_tree_2 = TreeContainer()
const_5 = 5 * np.ones(10)
const_4 = 4 * np.ones(10)
const_2 = 2 * np.ones(10)
testing_tree_2.ultimate_parent.add_left_child(TreeNode(False, None, addition, 2))
testing_tree_2.ultimate_parent.children[0].add_left_child(TreeNode(True, const_5))
testing_tree_2.ultimate_parent.children[0].add_right_child(TreeNode(False, None, division, 2))
testing_tree_2.ultimate_parent.children[0].children[1].add_left_child(TreeNode(True, const_4))
testing_tree_2.ultimate_parent.children[0].children[1].add_right_child(TreeNode(True, const_2))



print_subtree(testing_tree_2.ultimate_parent.children[0])



offspring_1, offspring_2 = crossover(testing_tree, testing_tree_2)

print("offspring_1")
print_subtree(offspring_1.ultimate_parent.children[0])
print("offspring_2")
print_subtree(offspring_2.ultimate_parent.children[0])