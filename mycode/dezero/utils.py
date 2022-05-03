import os
import subprocess
import pdb

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt +=dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # yはweakref
    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(lambda x: x.generation)
            seen_set.add(f)
    add_func(output.creator)
    txt += _dot_var(output, verbose)
    
    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)

        for x in f.inputs:
            txt += _dot_var(x)
        if x.creator is not None:
            add_func(x.creator)
    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # dotデータをファイルに保存
    parent_path = os.path.join(__file__, os.pardir) 
    tmp_dir = os.path.join(parent_path, 'tmp', '.dezero')
    to_file = os.path.join(tmp_dir, to_file)
    if not os.path.exists(tmp_dir): # tmp/.dezeroディレクトリがなかったら作成
        os.makedirs(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # dotコマンドを呼ぶ
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)