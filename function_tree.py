#%%
import random
import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
import time

class Node:
    def __init__(self, value, parent, kind):
        self.left_child = None
        self.right_child = None
        self.parent = parent
        self.value = value
        self.kind = kind

#counts number of pre-operators / functions within the equation
#use as: count_preops(function[find_root(function)], 0)
def count_preops(node, count):
    if node:
        if node.kind == 'P':
            count += 1
        if node.left_child:
            count = count_preops(node.left_child, count)
        if node.right_child:
            count = count_preops(node.right_child, count)
    return count

def count_consts(node, count):
    if node:
        if node.kind == 'C':
            count += 1
        if node.left_child:
            count = count_consts(node.left_child, count)
        if node.right_child:
            count = count_consts(node.right_child, count)
    return count

#measures time needed to compute the given function
def measure_comp_time(function, variable_list, C):
    var_scope = locals()
    readable = compute_function(function)
    for c in C:
        var_scope[c] = 2
    for v in variable_list:
        var_scope[v] = 2
    try:
        eval(readable)
        start = time.perf_counter()
        eval(readable)
        end = time.perf_counter()
        elapsed = end - start
        return elapsed
    except OverflowError:
        for c in C:
            var_scope[c] = -2
    except ZeroDivisionError:
        for c in C:
            var_scope[c] = -2
    try:
        eval(readable)
        start = time.perf_counter()
        eval(readable)
        end = time.perf_counter()
        elapsed = end - start
        return elapsed
    except OverflowError:
        return 10
    except ZeroDivisionError:
        return 10

def is_function(function):
    #check the given expression is a valid equation
    #TODO
    return True

def does_value_match_kind(function):
    #check the Node's value matches the Node's kind
    #TODO
    return True

def relabel_consts(function, new_const_symb):
    counter = 1
    tmp = relabel_const(function[find_root(function)], new_const_symb, counter)
    pass

def relabel_const(node, new_const_symb, counter):
    if node:
        counter = relabel_const(node.left_child, new_const_symb, counter)
        if node.kind == 'C':
            new_val = '' + new_const_symb + str(counter)
            node.value = new_val
            counter = counter + 1
        counter = relabel_const(node.right_child, new_const_symb, counter)
    return counter
    
def combine_consts(function, const_symb):
    function = combine_term(function, function[find_root(function)], const_symb)
    return function

def combine_term(function, node, const_symb):
    if node:
        function = combine_term(function, node.left_child, const_symb)
        function = combine_term(function, node.right_child, const_symb)
        if (node.kind == 'O') and (node.left_child.kind == 'C') and (node.right_child.kind == 'C'):
            if node.parent:
                par = node.parent
                new_node = Node(const_symb, par, 'C')
                if node == par.left_child:
                    par.left_child = new_node
                elif node == par.right_child:
                    par.right_child = new_node
                else:
                    return False
            else:
                new_node = Node(const_symb, None, 'C')
            function.remove(node)
            function.remove(node.left_child)
            function.remove(node.right_child)
            function.append(new_node)
            relabel_consts(function, const_symb)
        elif (node.kind == 'P') and (node.right_child.kind == 'C') and (node.right_child.right_child == None) and (node.right_child.left_child == None):
            if node.parent:
                par = node.parent
                new_node = Node(const_symb, par, 'C')
                if node == par.left_child:
                    par.left_child = new_node
                elif node == par.right_child:
                    par.right_child = new_node
                else:
                    return False
            else:
                new_node = Node(const_symb, None, 'C')
            function.remove(node)
            function.remove(node.right_child)
            function.append(new_node)
            relabel_consts(function, const_symb)
    return function

def find_root(function):
    root_index = None
    for i in range(0, len(function)):
        if function[i].parent == None:
            root_index = i
            break
    return root_index

def deepcopy(function, node):
    if node:
        value = node.value
        kind = node.kind
        if node.parent:
            parent = node.parent
        else:
            parent = None
        node_copy = Node(value, parent, kind)
        if function:
            function.append(node_copy)
        else:
            function = [node_copy]
        if node.left_child:
            function, left_child = deepcopy(function, node.left_child)
            node_copy.left_child = left_child
            left_child.parent = node_copy
        if node.right_child:
            function, right_child = deepcopy(function, node.right_child)
            node_copy.right_child = right_child
            right_child.parent = node_copy
    return function, node_copy

def compute_function(function):
    #prints function tree in a bottom - top, left - right manner
    root_index = find_root(function)
    if function:
        equation = compute_node('', function[root_index])
    else:
        equation = ''
    return equation

def compute_node(equation, node):
    if node:
        if (node.kind == 'V') or (node.kind == 'C'):
            equation = compute_node(equation, node.left_child)
            equation += str(node.value)
            equation = compute_node(equation, node.right_child)
        elif node.kind == 'O':
            equation += str('(')
            equation = compute_node(equation, node.left_child)
            equation += str(node.value)
            equation = compute_node(equation, node.right_child)
            equation += str(')')
        elif node.kind == 'P':
            equation += str('(')
            if node.value == 'pow2':
                equation += ('pow(')
                equation = compute_node(equation, node.right_child)
                equation += ', 2'
            elif node.value == 'pow3':
                equation += ('pow(')
                equation = compute_node(equation, node.right_child)
                equation += ', 3'
            elif node.value == 'pow4':
                equation += ('pow(')
                equation = compute_node(equation, node.right_child)
                equation += ', 4'
            elif node.value == 'pow5':
                equation += ('pow(')
                equation = compute_node(equation, node.right_child)
                equation += ', 5'
            else:
                equation += (str(node.value) + '(')
                equation = compute_node(equation, node.right_child)
            equation += str(')')
            equation += str(')')
    return equation

def remove_tree_from_function(function, node):
    if function:
        if node.parent:
            parent = node.parent
            if parent.left_child == node:
                parent.left_child = None
            elif parent.right_child == node:
                parent.right_child = None
        function = remove_node_from_function(function, node)
    return function

def remove_node_from_function(function, node):
    if node:
        function = remove_node_from_function(function, node.left_child)
        function = remove_node_from_function(function, node.right_child)
        function.remove(node)
    return function

def add_node_to_function(function, node):
    if node:
        if function:
            function.append(node)
        else:
            function = [node]
        function = add_node_to_function(function, node.left_child)
        function = add_node_to_function(function, node.right_child)
    return function

def mutation_alter(function, variable_list, P, C, O):
    #alter the term's operand or one of the terms symbols
    alter_index = random.randint(1, len(function)) - 1
    if (function[alter_index].kind == 'V') or (function[alter_index].kind == 'C'):
        v_or_c = random.randint(1, 2)
        if v_or_c == 1:
            new_value = C[0]
        else:
            if len(variable_list) > 1:
                new_value = variable_list[(random.randint(1, len(variable_list)) - 1)]
                while new_value == function[alter_index].value:
                    new_value = variable_list[(random.randint(1, len(variable_list)) - 1)]
            else:
                new_value = variable_list[0]
        function[alter_index].value = new_value
    elif function[alter_index].kind == 'O':
        if len(O) > 1:
            new_value = O[random.randint(1,len(O)) - 1]
            while new_value == function[alter_index].value:
                new_value = O[random.randint(1,len(O)) - 1]
            function[alter_index].value = new_value
    elif function[alter_index].kind == 'P':
        if len(P) > 1:
            new_value = P[random.randint(1,len(P)) - 1]
            while new_value == function[alter_index].value:
                new_value = P[random.randint(1,len(P)) - 1]
            function[alter_index].value = new_value
    else:
        print("Type mismatch: node neither variable or operator.")
        print(function[alter_index].kind)
    relabel_consts(function, C[0])
    combine_consts(function, C[0])
    return function

def mutation_reduce(function, C):
    if (len(function) > 1):
        red_index = random.randint(1, len(function)) - 1
        while (function[red_index].kind == 'V') or (function[red_index].kind == 'C'):
            red_index = random.randint(1, len(function)) - 1
        if function[red_index].parent:
            parent = function[red_index].parent
        else:
            parent = None
        if function[red_index].kind == 'O':
            l_or_r = random.randint(1, 2)
            if l_or_r == 1:
                function[red_index].left_child.parent = parent
                if parent:
                    if parent.left_child == function[red_index]:
                        parent.left_child = function[red_index].left_child
                    else: 
                        parent.right_child = function[red_index].left_child
                to_go = function[red_index].right_child
                function.remove(function[red_index])
                function = remove_node_from_function(function, to_go)
            else:
                function[red_index].right_child.parent = parent
                if parent:
                    if parent.left_child == function[red_index]:
                        parent.left_child = function[red_index].right_child
                    else:
                        parent.right_child = function[red_index].right_child
                to_go = function[red_index].left_child
                function.remove(function[red_index])
                function = remove_node_from_function(function, to_go)
        elif function[red_index].kind == 'P':
            function[red_index].right_child.parent = parent
            if parent:
                if parent.left_child == function[red_index]:
                    parent.left_child = function[red_index].right_child
                else: 
                    parent.right_child = function[red_index].right_child
            function.remove(function[red_index])
    relabel_consts(function, C[0])
    combine_consts(function, C[0])
    return function

def mutation_extend(function, variable_list, P, C, O):
    #add O S to the end of the term or apply a P to the whole term
    #find a random leaf that is of type variable
    extend_index = random.randint(1, len(function)) - 1
    while (function[extend_index].kind != 'V') and (function[extend_index].kind != 'C'):
        extend_index = random.randint(1, len(function)) - 1
    temp_parent = function[extend_index].parent
    mode = random.randint(1, 100)
    #mode 1 adds O S to a random operand and random variable node
    if mode < 80:
        if temp_parent:
            parent = function[extend_index].parent
        else:
            parent = None
        if len(O) > 1:
            new_operator = Node(O[random.randint(1,len(O)) - 1], parent, 'O')
        else:
            new_operator = Node(O[0], parent, 'O')
        if parent:
            if parent.left_child == function[extend_index]:
                parent.left_child = new_operator
            elif parent.right_child == function[extend_index]:
                parent.right_child = new_operator
        function[extend_index].parent = new_operator
        v_or_c = random.randint(1, 2)
        if v_or_c == 1:
            value = C[0]
            new_variable = Node(value, new_operator, 'C')
        else:
            if len(variable_list) > 1:
                value = variable_list[(random.randint(1, len(variable_list)) - 1)]
            else:
                value = variable_list[0]
            new_variable = Node(value, new_operator, 'V')
        l_or_r = random.randint(1, 2)
        if l_or_r == 1:
            new_operator.left_child = function[extend_index]
            new_operator.right_child = new_variable
        else:
            new_operator.left_child = new_variable
            new_operator.right_child = function[extend_index]
        function.append(new_operator)
        function.append(new_variable) 
    #mode 2 adds a pre-operator as parent of a random term
    else:
        if temp_parent:
            if len(P) > 1:
                new_preoperator = Node(P[random.randint(1,len(P)) - 1], function[extend_index].parent, 'P')
            else:
                new_preoperator = Node(P[0], function[extend_index].parent, 'P')
            new_preoperator.right_child = function[extend_index]
            if temp_parent.left_child == function[extend_index]:
                temp_parent.left_child = new_preoperator
            elif temp_parent.right_child == function[extend_index]:
                temp_parent.right_child = new_preoperator
        else:
            if len(P) > 1:
                new_preoperator = Node(P[random.randint(1,len(P)) - 1], None, 'P')
            else:
                new_preoperator = Node(P[0], None, 'P')
            new_preoperator.right_child = function[extend_index]
        function[extend_index].parent = new_preoperator
        function.append(new_preoperator)
    relabel_consts(function, C[0])
    combine_consts(function, C[0])
    return function

def cross_over(par_1, par_2, C):
    #Find the root elemnt for the parents
    root_index_p1 = find_root(par_1)
    root_index_p2 = find_root(par_2)
    #Deepcopy the parents to create the children. This creates duplicate nodes of each parent node
    child_1, tmp1 = deepcopy([], par_1[root_index_p1])
    child_2, tmp2 = deepcopy([], par_2[root_index_p2])
    relabel_consts(child_1, 'a')
    relabel_consts(child_2, 'b')
    #The parents need to be deepcopied too, to ensure children and parents are sorted in the same order
    parent_1, tmp_parent1 = deepcopy([], par_1[root_index_p1])
    parent_2, tmp_parent2 = deepcopy([], par_2[root_index_p2])
    relabel_consts(parent_1, 'a')
    relabel_consts(parent_2, 'b')
    #Choose the index to cross-over
    if len(parent_1) > 1:
        co_index_parent_1 = random.randint(1, len(parent_1)) - 1
    else:
        co_index_parent_1 = 0
    if len(parent_2) > 1:
        co_index_parent_2 = random.randint(1, len(parent_2)) - 1
    else:
        co_index_parent_2 = 0
    #Loop and re-choose index until we find acceptable elements to crossover. I.e.: We want to ensure proper sub-trees are copied so we only accept starts of terms to be swapped, which essentially is any variable or constant. We don't swap starting from pre-operators, as this could be a '-', which could be interpreted as an operator
    #Get the parent of the element to be swapped
    co_index_parent_1_parent = child_1[co_index_parent_1].parent
    co_index_parent_2_parent = child_2[co_index_parent_2].parent
    #Get the element to be swapped
    new_tree_start_1 = child_1[co_index_parent_1]
    new_tree_start_2 = child_2[co_index_parent_2]
    #Check if elements are root nodes, if not, check if we are swapping on the left or right side of the parent
    if (co_index_parent_1_parent):
        if (co_index_parent_1_parent.left_child) and (co_index_parent_1_parent.left_child == new_tree_start_1):
            tree_direction_1 = 'left'
        elif (co_index_parent_1_parent.right_child) and (co_index_parent_1_parent.right_child == new_tree_start_1):
            tree_direction_1 = 'right'
        else:
            tree_direction_1 = 'none'
    if (co_index_parent_2_parent):
        if (co_index_parent_2_parent.left_child) and (co_index_parent_2_parent.left_child == new_tree_start_2):
            tree_direction_2 = 'left'
        elif (co_index_parent_2_parent.right_child) and (co_index_parent_2_parent.right_child == new_tree_start_2):
            tree_direction_2 = 'right'
        else:
            tree_direction_2 = 'none'
    #First remove element and following sub-tree from the children, this alters the lists only, not the actual trees
    child_1 = remove_tree_from_function(child_1, new_tree_start_1)
    child_2 = remove_tree_from_function(child_2, new_tree_start_2)
    #Then add the element and following sub-tree from the other child, this alters the lists only, not the actual trees
    child_1 = add_node_to_function(child_1, new_tree_start_2)
    child_2 = add_node_to_function(child_2, new_tree_start_1)
    #Update the actual trees by updating the children-parent relationships
    new_tree_start_1.parent = co_index_parent_2_parent
    new_tree_start_2.parent = co_index_parent_1_parent
    if (co_index_parent_1_parent):
        if (tree_direction_1 == 'left'):
            co_index_parent_1_parent.left_child = new_tree_start_2
        elif (tree_direction_1 == 'right'):
            co_index_parent_1_parent.right_child = new_tree_start_2
    if (co_index_parent_2_parent):
        if (tree_direction_2 == 'left'):
            co_index_parent_2_parent.left_child = new_tree_start_1
        elif (tree_direction_2 == 'right'):
            co_index_parent_2_parent.right_child = new_tree_start_1
    relabel_consts(child_1, C[0])
    combine_consts(child_1, C[0])
    relabel_consts(child_2, C[0])
    combine_consts(child_2, C[0])
    return child_1, child_2

#Creates a single function with maximum length and list of variables (such as ['x']) as input
def create_function(maximum_length, variable_list, P, C, O):
    function_length = random.randint(1, maximum_length)
    v_or_c = random.randint(1, len(variable_list)+1)
    if v_or_c == 1:
        value = '' + C[0] + '1'
        root = Node(value, None, 'C')
    else:
        if len(variable_list) > 1:
            value = variable_list[(random.randint(1, len(variable_list)) - 1)]
        else:
            value = variable_list[0]
        root = Node(value, None, 'V')
    function = [root]
    for l in range(function_length):
        function = mutation_extend(function, variable_list, P, C, O)
    relabel_consts(function, C[0])
    combine_consts(function, C[0])
    return function
# %%
