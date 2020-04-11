
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

def add_node(parent, pl, lvl):
    pl = pl.strip()
    pos_node = pl.split('(')[1].strip()
    if len(pos_node) == 0 and not parent: # Recursive end
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

def main():
    test_text = '(ROOT\n  (S\n    (NP (NNS LoanNumbers))\n    (VP (MD may) (RB not)\n      (VP\n        (VP (VB be)\n          (NP (DT an) (JJ empty) (NN string)))\n        (CC or)\n        (VP\n          (ADVP (DT no) (JJR longer))\n          (PP (IN than)\n            (NP (CD 32) (NNS characters))))))))'
    print(test_text)
    ptree = ' '.join(test_text.split('\n')[1:])
    root = CNode('ROOT', 'ROOT', 0)
    add_node(root, ptree, 1)

    root.print_offsprings()


if __name__ == '__main__':
    main()
