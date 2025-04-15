import os
import torch
import torch.nn as nn
from gnn_data import GNN_DATA
from gnn_model import GIN_Net2
from utils import Metrictor_PPI

def evaluate(model, graph, batch_size, device):
    model.eval()
    val_mask = graph.val_mask
    steps = len(val_mask) // batch_size + int(len(val_mask) % batch_size != 0)
    
    preds, labels = [], []
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for step in range(steps):
            batch_ids = val_mask[step * batch_size : (step + 1) * batch_size]
            out = model(graph.x, graph.edge_index, batch_ids)
            label = graph.edge_attr_1[batch_ids]
            pred = (sigmoid(out) > 0.5).float()

            preds.append(pred.cpu())
            labels.append(label.cpu())
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    metrics = Metrictor_PPI(preds, labels)
    metrics.show_result(is_print=True)

def main():
    model_path = "models/gnn_random_2025-04-14_10-06-02.554155/gnn_model_valid_best.ckpt"  # change if stored elsewhere
    ppi_path = "data/protein.actions.SHS27k.STRING.txt"
    pseq_path = "data/protein.SHS27k.sequences.dictionary.tsv"
    vec_path = "data/vec5_CTC.txt"
    index_path = "data/train_valid_index_random.pkl"


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    ppi_data = GNN_DATA(ppi_path=ppi_path)
    ppi_data.get_feature_origin(pseq_path=pseq_path, vec_path=vec_path)
    ppi_data.generate_data()
    ppi_data.split_dataset(index_path, random_new=False, mode="random")

    graph = ppi_data.data
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']
    graph.to(device)

    model = GIN_Net2(in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    model.to(device)

    evaluate(model, graph, batch_size=64, device=device)

if __name__ == "__main__":
    main()
