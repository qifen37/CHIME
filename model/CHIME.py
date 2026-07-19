import torch.nn as nn
from model.decoder import Mlp


class CHIME(nn.Module):
    def __init__(self,
                 kg_g,
                 smiles,
                 hidden_dim,
                 num_layer,
                 class_num,
                 ):
        super(CHIME, self).__init__()

        self.smiles = smiles
        self.device = kg_g.device
        self.drug_num = len(smiles)
        self.hidden_dim = hidden_dim

        self.kg = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'], hidden_dim, num_layer)
        self.kg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.hyper_mol = HypergraphEncoder(smiles, hidden_dim, num_layer, self.device)

        self.interaction = SubstructureInteraction(hidden_dim, nhead=4, num_layers=2, dropout=0.1)
        self.decoder = Mlp(hidden_dim, hidden_dim, class_num=class_num)

    def forward(self, left, right):
        kg_emb = None
        kg_emb = self.kg()
        kg_emb = self.kg_proj(kg_emb)
        l_kg = kg_emb[left]
        r_kg = kg_emb[right]

        all_seqs, all_masks = self.hyper_mol(True)
        l_seq = all_seqs[left]
        l_mask = all_masks[left]

        r_seq = all_seqs[right]
        r_mask = all_masks[right]

        l_final, r_final = self.interaction(l_seq, l_mask, l_kg, r_seq, r_mask, r_kg)
 
        pred = self.decoder(l_final, r_final)

        return pred
