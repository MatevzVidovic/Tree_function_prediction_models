








import random

import numpy as np






class TreeContainer:

    def __init__(self):
        # ultimate parent isn't counted as a real node. It is only here so the root node also has a parent,
        #  which makes implementation easier.
        self.ultimate_parent = TreeNode(False, None, (pass_through, 1))
        self.list_of_nodes = list()
    
    def print(self):
        print_subtree(self.ultimate_parent.children[0])


    def copy(self):
        new_self = TreeContainer()
        new_self.ultimate_parent.children[0] = copy_subtree(self.ultimate_parent.children[0])
        new_self.ultimate_parent.children[0].parent = new_self.ultimate_parent
        append_nodes_from_subtree_to_list(new_self.ultimate_parent.children[0], new_self.list_of_nodes)

        return new_self
    
    def calculate(self):
        return self.ultimate_parent.children[0].calculate_subtree()
    
    def update_list_of_nodes(self):
        new_list = list()
        append_nodes_from_subtree_to_list(self.ultimate_parent.children[0], new_list)
        self.list_of_nodes = new_list
    

    def simple_mutation(self, possible_elem_func_tuples: list, possible_numeric_vectors: list):
        mutation_node = random.choice(self.list_of_nodes)

        if mutation_node.is_leaf:
            mutation_node.numeric_vector = random.choice(possible_numeric_vectors)
        else:
            new_elem_func_tuple = random.choice(possible_elem_func_tuples)
            
            if (mutation_node.elem_func_tuple[1] == new_elem_func_tuple[1]):
                mutation_node.elem_func_tuple = new_elem_func_tuple

            elif (mutation_node.elem_func_tuple[1] == 1 and new_elem_func_tuple[1] == 2):
                mutation_node.elem_func_tuple = new_elem_func_tuple

                # we create a new leaf node and assign it to the right child.
                chosen_numeric_vector = random.choice(possible_numeric_vectors)
                mutation_node.children[1] = TreeNode(True, chosen_numeric_vector)
                mutation_node.children[1].parent = mutation_node

                # we update the TreeContainer's list_of_nodes
                self.update_list_of_nodes()

            elif(mutation_node.elem_func_tuple[1] == 2 and new_elem_func_tuple[1] == 1):
                mutation_node.elem_func_tuple = new_elem_func_tuple
                # We decide to drop the right child.
                mutation_node.children[1] = None

                # we update the TreeContainer's list_of_nodes
                self.update_list_of_nodes()
            
            else:
                print("simple_mutation: the mutation couldn't happen.")
        
        return


    # This picks a node and inserts a new function node between it and it's parent.
    # If it inserts a function that takes two arguments it's right child will be a new leaf.
    # def inserting_mutation(self, possible_elem_func_tuples: list, possible_numeric_vectors: list):
    #     mutation_node = random.choice(self.list_of_nodes)
    #     parent_node = mutation_node.parent
    #     new_elem_func_tuple = random.choice(possible_elem_func_tuples)
    #     new_function_node = TreeNode(False, None, new_elem_func_tuple)

    #     # This function will not be finished right now, since it isn't necessary.
        




        









class TreeNode:

    # The function you pass has to ba a tuple of the form: (function, num_of_args_of__this_function)
    def __init__(self, is_leaf=False, numeric_vector=None, elem_func_tuple=(None, 0)):
        self.is_leaf = is_leaf
        self.numeric_vector = numeric_vector

        self.elem_func_tuple = elem_func_tuple

        self.parent = None
        self.children = [None, None]
    

    def calculate_subtree(self):
        if(self.is_leaf):
            return self.numeric_vector
        elif(self.elem_func_tuple[1] == 1):
            return self.elem_func_tuple[0](self.children[0].calculate_subtree())
        elif(self.elem_func_tuple[1] == 2):
            return self.elem_func_tuple[0](self.children[0].calculate_subtree(), self.children[1].calculate_subtree())
        else:
            print("This (" + str(self) + ") is not a leaf and has " + str(self.elem_func_tuple[1]) + "as elem_func_tuple[1], which is different from 1 or 2.")
    

    def copy_without_parent_and_children(self):
        new_self = TreeNode(self.is_leaf, self.numeric_vector, self.elem_func_tuple)
        return new_self
    
    def set_left_child(self, adding_node):
        self.children[0] = adding_node
        adding_node.parent = self
        return
    
    def set_right_child(self, adding_node):
        self.children[1] = adding_node
        adding_node.parent = self
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


        # selects a random element of the list
        exchanged_node_1 = random.choice(parent_tree_1.list_of_nodes)
        exchanged_node_2 = random.choice(parent_tree_2.list_of_nodes)

        
        parent_node_1 = exchanged_node_1.parent
        parent_node_2 = exchanged_node_2.parent


        # We have to find which of the parent's children is being replaced, and replace it with the new node.
        # e.g. parent_node_1 has two children. We need to find out which of these two children is exchange_node_1.
        # This child will then be replaced with exchange_node_2.
        if (parent_node_1.children[0] == exchanged_node_1):
            parent_node_1.children[0] = exchanged_node_2
        else:
            parent_node_1.children[1] = exchanged_node_2

        # We do the same for parent_node_2 and exchange_node_2     
        if (parent_node_2.children[0] == exchanged_node_2):
            parent_node_2.children[0] = exchanged_node_1
        else:
            parent_node_2.children[1] = exchanged_node_1
        

        # We also correct the parents of our exchanged nodes.
        exchanged_node_1.parent = parent_node_2
        exchanged_node_2.parent = parent_node_1


        # We empty the lists of nodes of our new trees, and we build them again.
        parent_tree_1.list_of_nodes = list()
        parent_tree_2.list_of_nodes = list()
        append_nodes_from_subtree_to_list(parent_tree_1.ultimate_parent.children[0], parent_tree_1.list_of_nodes)
        append_nodes_from_subtree_to_list(parent_tree_2.ultimate_parent.children[0], parent_tree_2.list_of_nodes)


        return parent_tree_1, parent_tree_2

def print_subtree(subtree_root: TreeNode, level_tuple=()):

    if subtree_root == None:
        return

    if subtree_root.is_leaf:
        print(str(level_tuple) + ":")
        print(subtree_root.numeric_vector)
    else:
        print(str(level_tuple) + ":")
        print(subtree_root.elem_func_tuple)
    
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

def cosine(first_array):
    return_array = np.cos(first_array)
    return return_array







# Unit test

# Creating our first testing_tree:

testing_tree = TreeContainer()
const_5 = 5 * np.ones(10)
const_4 = 4 * np.ones(10)
const_2 = 2 * np.ones(10)
testing_tree.ultimate_parent.set_left_child(TreeNode(False, None, (addition, 2)))
testing_tree.ultimate_parent.children[0].set_left_child(TreeNode(True, const_5))
testing_tree.ultimate_parent.children[0].set_right_child(TreeNode(False, None, (division, 2)))
testing_tree.ultimate_parent.children[0].children[1].set_left_child(TreeNode(True, const_4))
testing_tree.ultimate_parent.children[0].children[1].set_right_child(TreeNode(True, const_2))

print("testing_tree was created and we will test units on it.")
print("testing_tree:")
testing_tree.print()

print("\n\n\n")
print("Result of testing_tree.calculate (should be 7): " + str(testing_tree.calculate()))

# print(testing_tree.ultimate_parent.children[0].calculate_subtree())
# print(testing_tree.ultimate_parent.calculate_subtree())

# print(testing_tree.ultimate_parent.children[0].children[1].calculate_subtree())


testing_list = list()
append_nodes_from_subtree_to_list(testing_tree.ultimate_parent.children[0].children[1], testing_list)

print("\n\n\n")
print("len of testing_list from: append_nodes_from_subtree_to_list(testing_tree.ultimate_parent.children[0].children[1], testing_list)")
print("Len should be 3: " + str(len(testing_list)))
# print("\n\n\n")
# print("And the resulting testing_list itself:")
# print(testing_list)


testing_list = list()
append_nodes_from_subtree_to_list(testing_tree.ultimate_parent.children[0].children[0], testing_list)
print("\n\n\n")
print("len of testing_list from: append_nodes_from_subtree_to_list(testing_tree.ultimate_parent.children[0].children[0], testing_list)")
print("Len should be 1: " + str(len(testing_list)))
# print("\n\n\n")
# print("And the resulting testing_list itself:")
# print(testing_list)


testing_list = list()
append_nodes_from_subtree_to_list(testing_tree.ultimate_parent.children[0], testing_list)
print("\n\n\n")
print("len of testing_list from: append_nodes_from_subtree_to_list(testing_tree.ultimate_parent.children[0], testing_list)")
print("Len should be 5: " + str(len(testing_list)))
# print("\n\n\n")
# print("And the resulting testing_list itself:")
# print(testing_list)






testing_tree_2 = TreeContainer()
const_5 = 5 * np.ones(10)
const_4 = 4 * np.ones(10)
const_2 = 2 * np.ones(10)
testing_tree_2.ultimate_parent.set_left_child(TreeNode(False, None, (addition, 2)))
testing_tree_2.ultimate_parent.children[0].set_left_child(TreeNode(True, const_5))
testing_tree_2.ultimate_parent.children[0].set_right_child(TreeNode(False, None, (division, 2)))
testing_tree_2.ultimate_parent.children[0].children[1].set_left_child(TreeNode(True, const_4))
testing_tree_2.ultimate_parent.children[0].children[1].set_right_child(TreeNode(True, const_2))

print("\n\n\n")
print("testing_tree_2:")
testing_tree_2.print()



offspring_1, offspring_2 = crossover(testing_tree, testing_tree_2)

print("\n\n\n")
print("We conducted a crossover between testing_tree_1 and testing_tree_2 (which are identical).")
print("These are the resulting offsprings:")


print("\n\n\n")
print("offspring_1")
offspring_1.print()

print("\n\n\n")
print("offspring_2:")
offspring_2.print()

print("\n\n\n")
print("Calculation on the offspring_2 tree:")
print(offspring_2.calculate())





print("\n\n\n")
print("We want to test mutations. We conduct a mutation on the offspring_2 tree.")

testing_mutation_function_tuples = [(addition, 2), (subtraction, 2), (multiplication, 2), (division, 2), (exponentiation, 2), (cosine, 1)]
base_vector = np.ones(10)
testting_numeric_vectors = [2 * base_vector, 3*base_vector, 5*base_vector]

offspring_2.simple_mutation(testing_mutation_function_tuples, testting_numeric_vectors)



print("\n\n\n")
print("List of possible function tuples: " + str(testing_mutation_function_tuples))
print()
print("offspring_2:")
offspring_2.print()

print(offspring_2.calculate())



print("\n\n\n")
print("Continued mutation with cosine being the only possible function:")

testing_mutation_function_tuples = [(cosine, 1)]

offspring_2.simple_mutation(testing_mutation_function_tuples, testting_numeric_vectors)

print("offspring_2:")
print()
offspring_2.print()

print(offspring_2.calculate())
