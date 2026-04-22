# README

The files in the `Neo4J Node` folder are mainly used for knowledge graph construction.

The contents of this folder can be broadly divided into two parts.

The first part consists of the files related to `doc.ipynb`. These files include a selection of example articles, as well as the scripts used to construct the knowledge graph based on the content of these articles. In particular, these scripts are used to extract vocabulary, concepts, and their relationships from the articles, and then generate the corresponding nodes and edges for further import into Neo4j or for subsequent knowledge graph analysis. This part mainly demonstrates how the knowledge graph is built step by step from specific text materials at the article level.

The second part is the `neo4j_script.ipynb` file. This file is mainly used to construct the document-level knowledge graph for the overall **Yanyuwa** corpus. Compared with the first part, which focuses more on node construction from individual articles or local text samples, this file is designed to organize and connect information at a higher level across the full set of documents, thereby forming a broader knowledge graph structure for the project.
