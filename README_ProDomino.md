# ProDomino: Rational engineering of allosteric protein switches by *in silico* prediction of domain insertion sites

<img src="img/ProDomino.png" alt="drawing" width="300"/>  

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/

[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Abstract 

Domain insertion engineering is a powerful approach to juxtapose otherwise separate biological functions, resulting in proteins with new-to-nature activities. A prominent example are switchable protein variants, created by receptor domain insertion into effector proteins. Identifying suitable, allosteric sites for domain insertion, however, typically requires extensive screening and optimization.
We present ProDomino, a novel machine learning pipeline to rationalize domain recombination, trained on a semi-synthetic protein sequence dataset derived from naturally occurring intradomain insertion events. ProDomino robustly identifies domain insertion sites in proteins of biotechnological relevance, which we experimentally validated in E. coli and human cells. Finally, we employed light- and chemically regulated receptor domains as inserts and demonstrate the rapid, model-guided creation of potent, single-component opto- and chemogenetic protein switches. These include novel CRISPR-Cas9 and -Cas12a variants for inducible genome engineering in human cells. Our work enables one-shot domain insertion engineering and substantially accelerates the design of customized allosteric proteins.


---

ProDomino enables the prediction of domain insertion sites  The model returns a per position probability score for insertion site tolerance.
For further details, please refer to our manuscript: https://www.biorxiv.org/content/10.1101/2024.12.04.626757v1

## Requirements

All experiments were run using python 3.10 and pytorch 2.10 using CUDA 12.1 and CUDNN 8.9.2 on Red Hat Enterprise Linux
8.10  
For insertion site prediction, ProDomino uses embeddings from the [ESM-2](https://github.com/facebookresearch/esm) 3B model.

## Installation

To use this repo make sure tha you have installed conda or mamba on your device.
Then run:
```bash
conda env create -n prodomino --file environment.yml
```

Installation should take 10 - 15 minutes.  
A likely point of installation failure are CUDA version conflicts. If installation fails install the required CUDA or adapt them to your systems version.


## Usage

ProDomino provides two main classes: `Embedder` and `ProDomino`.

`Embedder` provides an interface to ESM-2 to generate the required input data for `ProDomino`

`ProDomino` then generates the final prediction and provides various plotting utilities.

```python
from ProDomino import Embedder, ProDomino

seq = ''

embedder = Embedder()
model = ProDomino(
    chkpt,'mini_3b_mlp')

embed = embedder.predict_embedding(seq)
pred = model.predict_insertion_sites(embed)
```

Runtime will differ based on the input sequence but should be under 5 minutes using a GPU.  
A complete example can be found in `example.ipynb`




---
## Cite




