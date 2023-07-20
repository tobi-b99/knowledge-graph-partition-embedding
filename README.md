# Knowledge Graph Embedding Partition

The goal of this project is to experiment with partitioning knowledge graphs (KG) before they are embedded and evaluating the performance differences compared to using one big KG.


## Dependencies

The project was done using Python 3.7.12 with following required depencies:
- zlib (tested with 1.2.13)
- numpy==1.19.1
- pandas==1.0.5
- scipy==1.5.1
- scikit-learn==0.23.1
- h5py==2.10.0
- pathlib2 (tested with 2.3.7.post1)
- requests (tested with 2.28.1)
- tqdm (tested with 4.64.1)
- gensim (tested with 3.8.3)
- aiohttp (tested with 3.8.4)
- evaluation-framework==2.0


### Installing Evaluation-Framework ([GEval](https://github.com/mariaangelapellegrino/Evaluation-Framework))

Since the project is not up to date on [pypi.org](pypi.org) (at the time of this project), either the dependencies have to be downgraded to fit an older version or a newer version can be installed directly from Github for example like this:
```
pip install git+https://github.com/mariaangelapellegrino/Evaluation-Framework@master
```
### Partitioning
For partitioning an [adaptation](https://github.com/dice-group/rdf-partitioning) of [Koral](https://github.com/Institute-Web-Science-and-Technologies/koral) by Akther et al. was used to partition and encode the graph into chunks. Then the ChunkTranslator provided by Koral was used to transform the chunks back into decoded graph files. See the READMEs of the corresponding projects for more information.

### jRDF2vec
[jRdf2Vec](https://github.com/dwslab/jRDF2Vec), an implementation of RDF2vec in Java, was used for embedding the graphs. First the walks were created, then the embeddings for the graphs.

## Dataset
The dataset that was used is [Cleaned object properties extracted with mappings](https://databus.dbpedia.org/dbpedia/mappings/mappingbased-objects).
