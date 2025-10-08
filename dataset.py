# dataset.py
from utils import get_name, get_data, get_embedding
from graph_process.molecule_graph import get_molecules_graph
from graph_process.protein_graph import get_proteins_graph

def load(datapath, dataname):
    dataset = get_data(dataname, datapath)
    molecules_graph = get_molecules_graph()
    protein_graph = get_proteins_graph()
    molecules_embedding, protein_embedding, gene_embedding = get_embedding()
    return dataset, molecules_graph, protein_graph, molecules_embedding, protein_embedding, gene_embedding

# 加载数据集
def load_dataset(dataname):
    datapath = './datasets/' + dataname + '/'
    return load(datapath, dataname)


# datename = 'Luo'
# nfold = 10
# neg_pos_ratio = 10
# seed_cross = 42
# f = load_dataset(datename, nfold, neg_pos_ratio, seed_cross)