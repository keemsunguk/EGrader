
class CNode:
    def __init__(self, ntype, name, depth, parent=None):
        self.node_type = ntype
        self.name = name
        self.children = []
        self.depth = depth
        self.parent = parent

    def __str__(self):
        return self.name+": "+str(len(self.children))+" leaves at " + str(self.depth)

    def print_offsprings(self):
        print('-'*self.depth, self)
        if len(self.children) > 0:
            for c in self.children:
                c.print_offsprings()


def travel_tree(node, path, node_list=[]):
    if len(path) == 0:  # Stop recursive
        return
    if path[0] == node.name:
        if len(path) > 1:
            found = False
            for c in node.children:
                if path[1] == c.name:
                    found = True
                    travel_tree(c, path[1:], node_list)
            if not found:
                print('Children not found')
                return None
        else:
            node_list.append(node)
            return
    else:
        print('Not found:', path[0])
        return


def get_words(node, phrase):
    """
    Get all the terminal node words
    """
    if not node:
        return phrase
    if node.node_type == 'TOKEN':
        phrase.append(node.name)
        return phrase
    else:
        for c in node.children:
            phrase = get_words(c, phrase)
        return phrase


def get_phrase_list(root, term_path):
    phrase_list = []
    if type(term_path[0]) == list:
        for tp in term_path:
            term_node_list = []
            travel_tree(root, tp, term_node_list)
            for t_node in term_node_list:
                term = get_words(t_node, [])
                phrase_list.append(' '.join(term))
        return phrase_list
    else:
        term_node_list = []
        travel_tree(root, term_path, term_node_list)
        for t_node in term_node_list:
            term = get_words(t_node, [])
            phrase_list.append(' '.join(term))
        return phrase_list

def add_node(parent, pl, lvl):
    pl = pl.strip()
    pos_node = pl.split('(')[1].strip()
    if len(pos_node) == 0 and not parent:  # Recursive end
        return
    up_level = 0
    ntype = 'POS'
    if len(pos_node.split()) == 2:
        pos_node = pos_node.split()[0]
        sub_tree = '('+pl[1+len(pos_node)+1:]
    else:
        sub_tree = '(' + '('.join(pl.split('(')[2:])
    if ')' in pos_node:
        up_level = pos_node.count(')')
        pos_node = pos_node.split(')')[0]
        ntype = 'TOKEN'
    node = CNode(ntype, pos_node, lvl, parent)
    parent.children.append(node)
    for i in range(up_level):
        parent = parent.parent
    if up_level > 0:
        add_node(parent, sub_tree, lvl - up_level)
    else:
        add_node(node, sub_tree, lvl + 1)

boundaries = {
    'longer than': '>',
    'no longer than': '<=',
    'shorter than': '<',
    'equal': '==',
    'is': '==',
    'be': '=='
}

def generate_python(term, negate, conjunction, definiends):
    template = '''
def test_^term^(input_str):
    if len(input_str) ^lower_bound^ ^conjunction^ len(input_str) ^upper_bound^:
        return ^return_val^
    else:
        return not ^return_val^
'''
    lower_bound = ''
    upper_bound = ''
    for k, v in boundaries.items():
        if k in definiends[0]:
            lower_bound = v
        if k in definiends[1]:
            upper_bound = v
    lb = definiends[0].split()
    ub = definiends[1].split()
    for tok in lb:
        tok = '0' if tok == 'empty' else tok
        if tok.isnumeric():
            lower_bound += tok
    for tok in ub:
        tok = '0' if tok == 'empty' else tok
        if tok.isnumeric():
            upper_bound += tok

    return_val = 'False' if 'not' in negate else 'True'
    template = template.replace('^term^', term)
    template = template.replace('^lower_bound^', lower_bound)
    template = template.replace('^conjunction^', conjunction)
    template = template.replace('^upper_bound^', upper_bound)
    template = template.replace('^return_val^', return_val)

    return template


def main():
    test_text = '(ROOT\n  (S\n    (NP (NNS LoanNumbers))\n    (VP (MD may) (RB not)\n      (VP\n        (VP (VB be)\n          (NP (DT an) (JJ empty) (NN string)))\n        (CC or)\n        (VP\n          (ADVP (DT no) (JJR longer))\n          (PP (IN than)\n            (NP (CD 32) (NNS characters))))))))'
    print(test_text)
    ptree = ' '.join(test_text.split('\n')[1:])
    root = CNode('ROOT', 'ROOT', 0)
    add_node(root, ptree, 1)

    root.print_offsprings()

    term_path = ['ROOT', 'S', 'NP']
    node_list = []
    travel_tree(root, term_path, node_list)
    for node in node_list:
        phrase_list = get_words(node, [])
        print(' '.join(phrase_list))

    node_list = []
    verb_path = ['ROOT', 'S', 'VP', 'VP', 'VP']
    travel_tree(root, verb_path, node_list)
    for v_node in node_list:
        verb = get_words(v_node, [])
        print(' '.join(verb))

    term_path = ['ROOT', 'S', 'NP']
    definiend_path = ['ROOT', 'S', 'VP', 'VP', 'VP']
    verb_path = ['ROOT', 'S', 'VP', 'RB']
    conj_path = ['ROOT', 'S', 'VP', 'VP', 'CC']

    terms = get_phrase_list(root, term_path)
    verbs = get_phrase_list(root, verb_path)
    definiends = get_phrase_list(root, definiend_path)
    conjunction = get_phrase_list(root, conj_path)
    print(terms[0])
    print(verbs[0])
    print(definiends)
    print(conjunction[0])
    print(generate_python(terms[0], verbs[0], conjunction[0], definiends))


if __name__ == '__main__':
    main()
