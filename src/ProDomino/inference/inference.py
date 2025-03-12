import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..models import MODELS
from ..utils.tracker import Hparams

import esm
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import PDBParser, PDBIO
from io import StringIO
import py3Dmol
class Embedder:
    def __init__(self):
        url = "tcp://localhost:8891"
        torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)
        model_name = "esm2_t36_3B_UR50D"
        model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)
        # initialize the model with FSDP wrapper
        fsdp_params = dict(
            mixed_precision=True,
            flatten_parameters=True,
            state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
            cpu_offload=True,  # enable cpu offloading
        )
        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            model, vocab = esm.pretrained.load_model_and_alphabet_core(
                model_name, model_data, regression_data
            )
            self.batch_converter = vocab.get_batch_converter()
            model.eval()

            # Wrap each layer in FSDP separately
            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)
            self.model = wrap(model)
    def predict_embedding(self,sequence,name=None):
        name = 'Sequence' if name is None else name
        print(len(sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter([(name,sequence)])
        batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[36], return_contacts=True)
        token_representations = results["representations"][36][0, 1:-1]
        return token_representations.cpu().numpy().squeeze().astype(np.float32)

    def batch_predict_embedding(self,processing_list):
        """
        processing_list: list = [[(name,sequence)],[...],[(name,sequence)]]
        """
        proceesed_results = {}
        for data in processing_list:
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.cuda()
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[36], return_contacts=True)
            token_representations = results["representations"][36][0, 1:-1].cpu().numpy().squeeze().astype(np.float32)
            proceesed_results[batch_labels] = token_representations
        return proceesed_results

class ProDomino():
    def __init__(self,model_weights,model_type,device=None):
        self.device = device
        self.model_weights = model_weights
        self.model_type = model_type
        hparams = Hparams({'model':{'optimizer':{'name':'adam','learning_rate':1e-3}}})

        self.model = MODELS[model_type].load_from_checkpoint(model_weights,
                                                        input_channels=2560,
                                                        optimizer=hparams.model.optimizer, loss='bce',
                                                        map_location=torch.device('cpu'))
        if self.device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()
    def predict_insertion_sites(self,esm_embedding):
        input_tensor = torch.from_numpy(np.expand_dims(esm_embedding,0)).float().to(self.device)
        with torch.no_grad():
            predicted_sites = self.model(input_tensor).cpu()

        return InsertionSitePrediction(torch.sigmoid(predicted_sites).squeeze(),esm_embedding)


class InsertionSitePrediction:
    def __init__(self,predicted_sites,esm_embedding,sequence=None,pdb_path=None):
        self.esm_embedding = esm_embedding
        self.predicted_sites = predicted_sites
        self.sequence = sequence

        if pdb_path is not None:
            par = PDBParser()
            struct = par.get_structure('', pdb_path)
            self.pdb = struct


    def show_trace(self,show_top_hits=False,linebreak=300,n_top_hits=10):
        if self.sequence is not None:
            rows = (len(self.sequence) // linebreak) + 1
        else:
            rows = (len(self.predicted_sites) // linebreak) + 1
        fig, axs = plt.subplots(rows, 1, figsize=(11, 1 * rows), dpi=300)
        if rows == 1:
            axs = [axs] # This ensures that the indexing still works
        if show_top_hits:
            for tophit in self.get_top_hits(n_top_hits):
                axs[(tophit // linebreak)].vlines(tophit % linebreak, 0, self.predicted_sites[tophit], color='lime')

        for nr, pos in enumerate(range(0, len(self.predicted_sites), linebreak)):
            sns.lineplot(self.predicted_sites[pos:pos + linebreak], color='black', ax=axs[nr], )
            axs[nr].set_ylim(0, 1, )
            axs[nr].set_xlim(0, linebreak)

            if self.sequence is not None:
                seq_subset = list(self.sequence)[pos:pos + linebreak]
                if len(seq_subset) < linebreak:
                    seq_subset.extend([' ' for i in range(linebreak - len(seq_subset))])
                axs[nr].set_xticks(range(0, linebreak), seq_subset, fontsize=5)
                sec = axs[nr].secondary_xaxis(location=0)
                sec.set_xticks([i for i in range(0,linebreak,10)], labels=[f'\n{i + (nr*linebreak)}' for i in range(1,linebreak+1,10)])

        fig.suptitle(f'ProDomino Prediction')
        fig.tight_layout()
        plt.show()

    def get_top_hits(self,n=10):
        top10 = np.argpartition(self.predicted_sites.squeeze().numpy(), -n)[-n:]
        top10.sort()
        return top10

    def add_gt_data(self):
        pass
    def add_pdb_file(self,pdb_file,chain_id='A',shift=0):
        self.shift = shift
        self.chain_id = chain_id
        par = PDBParser()
        struct = par.get_structure(chain_id, pdb_file)
        self.pdb = struct
        if self.sequence is None:
            return

        align_arr = np.zeros(len(self.sequence), dtype=object)
        align_arr[:] = ' '
        for model in struct:
            for chain in model:
                if chain.id == self.chain_id:
                    for nr, res in enumerate(chain):
                        if res.get_id()[1] - 1 < len(self.sequence):
                            try:
                                aa = three_to_one(res.get_resname())
                            except KeyError:
                                aa = 'X'
                            align_arr[res.get_id()[1] - 1] = aa

        pdb_str = ''.join(align_arr)
        for i in range(0, len(self.sequence), 120):
            pdb_sub = self.sequence[i:i + 120]
            if shift >= 0 or i != 0:
                seq_sub = pdb_str[i + shift:i + 120 + shift]
            else:
                seq_sub = -shift * '_' + pdb_str[i:i + 120 + shift]
            print(f'SEQ: {pdb_sub}')
            print(f'PDB: {seq_sub}\n')
    def add_sequence(self,sequence):
        self.sequence = sequence

    def generate_insertion_site_pdb_file(self):
        array = np.round(self.predicted_sites, 2) * 100
        for model in self.pdb:
            for chain in model:
                if chain.id == self.chain_id:
                    for nr, res in enumerate(chain):
                        if res.get_id()[1] - 1 - self.shift < len(self.sequence):
                            try:
                                pdb_aa = three_to_one(res.get_resname())
                            except KeyError:
                                pdb_aa = 'X'
                            if self.sequence[res.get_id()[1] - 1 - self.shift] == pdb_aa:
                                for atom in res.get_atoms():
                                        if not res.id[1]-1 > len(array):
                                            value = array[res.id[1]-1- self.shift]
                                            atom.set_bfactor(value)

    def show_pdb(self):
        self.generate_insertion_site_pdb_file()
        pdbio = PDBIO()
        pdbio.set_structure(self.pdb)
        pdb_string_io = StringIO()
        pdbio.save(pdb_string_io)
        raw_pdb_data = pdb_string_io.getvalue()
        p = py3Dmol.view(data=raw_pdb_data,width=800,height=800)
        p.setStyle({"cartoon": {'color': 'grey','opacity':0.5}})
        p.setStyle({"chain": self.chain_id},
                   {"cartoon": {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 0, 'max': 100,'opacity':1}}})
        p.show()

    def compute_sasa(self):
        sr = ShrakeRupley()
        sr.compute(self.pdb, level="R")
        sasa_arr = np.zeros(len(self.sequence))
        sasa_arr[:] = np.nan
        for model in self.pdb:
            for chain in model:
                if chain.id == self.chain_id:
                    for nr, res in enumerate(chain):
                        if res.get_id()[1] - 1 - self.shift < len(self.sequence):
                            try:
                                pdb_aa = three_to_one(res.get_resname())
                            except KeyError:
                                pdb_aa = 'X'
                            if self.sequence[res.get_id()[1] - 1 - self.shift] == pdb_aa:
                                sasa_arr[res.get_id()[1] - 1 - self.shift] = res.sasa
        return sasa_arr
    def show_sasa_plot(self):
        sasa_arr = self.compute_sasa()
        plt.figure(figsize=(4,4))
        sns.scatterplot(x=sasa_arr,y=self.predicted_sites)
        plt.xlabel('SASA [ShrakeRupley]')
        plt.ylabel('Predicted insertion tolerance')