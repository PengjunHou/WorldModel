# Reproducing the world model with pytorch

export env settings: conda env export --no-builds > environment.yml

configure env: conda env create -f environment.yml

blog: https://blog.otoro.net//2018/06/09/world-models-experiments/

github: https://github.com/PengjunHou/WorldModelsExperiments#

paper:
```latex
@incollection{ha2018worldmodels,
  title = {Recurrent World Models Facilitate Policy Evolution},
  author = {Ha, David and Schmidhuber, J{\"u}rgen},
  booktitle = {Advances in Neural Information Processing Systems 31},
  pages = {2451--2463},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution},
  note = "\url{https://worldmodels.github.io}",
}
```