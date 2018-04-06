import numpy as np
# a basic decision tree


def partition(node, question):  # nodes are made up of rows of data
    true_rows, false_rows = [], []
    for row in node:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def ask_all_questions(node):
    uniques = []
    for i in range(len(header) - 1):
        a = unique_set(node, i)  # prevents asking homogeneous nodes OR nodes with only one feature value
        a = list(a)
        uniques.append(a)
    best_question = None
    lowest_disorder = 1
    for j in range(len(uniques)):
        for subject in uniques[j]:
            question = Question(j, subject)
            true_rows, false_rows = partition(node, question)
            current_disorder = nodes_disorder(true_rows, false_rows)
            if current_disorder < lowest_disorder:
                lowest_disorder = current_disorder
                best_question = question
    return lowest_disorder, best_question


def build_tree(rows):
    lowest_disorder, best_question = ask_all_questions(rows)
    if best_question is None:  # or lowest_disorder == 1:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, best_question)
    print(true_rows, "AND", false_rows)  # remove [3] in both to see their respective details
    true_node = build_tree(true_rows)
    false_node = build_tree(false_rows)
    return DecisionNode(true_node, false_node, best_question)


class Question:
    def match(self, row):
        if row[self.column] == self.value:
            return True
        else:
            return False

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def __repr__(self):
        return "Is %s == %s ?" % (header[self.column], self.value)


class Leaf:
    def __init__(self, rows):
        self.predictions = label_count(rows)

    def __repr__(self):
        return self.predictions


class DecisionNode:
    def __init__(self, true_rows, false_rows, question):
        self.true_rows = true_rows
        self.false_rows = false_rows
        self.question = question


def nodes_disorder(true_rows, false_rows):
    total_rows = len(true_rows) + len(false_rows)
    return disorder(true_rows) * len(true_rows) / total_rows + disorder(false_rows) * len(false_rows) / total_rows


def disorder(rows):
    labels_density = label_count(rows)
    total_items = 0
    dis = 0
    for val in labels_density.values():
        total_items += val
    densities = []
    for label in labels_density.keys():
        densities.append(labels_density[label])
    terms = len(unique_set(rows, 2))
    if terms > 1:
        for i in range(terms):
            nom = densities[i]
            dis += - nom / total_items * log(terms, nom / total_items)  # borrowed from MIT lectures
    return dis


def unique_set(rows, col):  # set for the unique elements of a column given the rows
    a = set([row[col] for row in rows])
    b = set([row[2] for row in rows])
    if len(a) == 1 or len(b) == 1:  # makes sure we don't ask questions for only one type of a fruit later on
        return set([])
    else:
        return a
    # return set([row[col] for row in rows])


def label_count(rows):  # dict for the density of fruits
    class_count = {}
    for row in rows:
        label = row[2]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count


def log(base, val):  # for later on if we want to make non binary questions
    if val <= 0 or base == 0 or base == 1:  # in that case the base will be greater than 2
        return 0
    else:
        return np.log2(val) / np.log2(base)


data = [
    ['heavy', 1, 1, 'Vampire'],  # classic vampire but the fact that he likes garlic
    ['odd', 0, 0, 'Vampire'],  # a vampire that has adapted his accent and got a tan
    ['heavy', 1, 0, 'Vampire'],  # classic vampire
    ['odd', 1, 1, 'Normal'],  # a pale, garlic loving local
    ['odd', 0, 1, 'Normal'],  # a random, normal looking local
    ['odd', 1, 1, '?'],
    ['heavy', 0, 2, '?']  # just a person that REALLY loves garlic
]
header = ["accent", "pale", "garlic", "label"]  # 1 notates yes and 0 notates no

build_tree(data)
