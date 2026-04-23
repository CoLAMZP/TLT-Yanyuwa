# README

The files in the `Neo4J Node` folder are mainly used for knowledge graph construction.

The contents of this folder can be broadly divided into two parts.

The first part consists of the Jupyter Notebook files ending with `_Doc.ipynb` or `_doc{Number}_word.ipynb`. The files ending with `_Doc.ipynb` are mainly used to organise source materials and create nodes and relationships for article-level knowledge graphs based on individual articles. The files ending with `_doc{Number}_word.ipynb` contain the selected example articles, together with information about researcher relationships and their associated data. Together, these files show how article-level knowledge graph components are prepared from source materials and supporting research information.

The second part is the `neo4j_script.ipynb` file. Similar to the `_Doc.ipynb` files, this file is also used to organise data and create nodes and relationships. However, instead of focusing on individual articles, it is designed to build the document-level knowledge graph for the overall **Yanyuwa** corpus. In this way, it connects information across the full set of documents and supports the construction of a broader project-level knowledge graph structure.

`YanyuwaProjectExperimentCodes` foldfer contains the code for the **Part-of-Speech Tagging** and **Named Entity Recognition** tasks. It includes the overall training pipeline, experimental results, and the relevant environment settings and dependencies required for reproducing the experiments.
