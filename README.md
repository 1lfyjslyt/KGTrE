# KGTrE

```
KGTrE: Knowledge Graph Recommendation based on Tree-Structure Embedding
```

## Basic requirements

* python 3.10.9
* tensorflow 2.10.0
* numpy 1.23.5
* scikit-learn1.2.1
* torch2.0.0
* pandas1.5.3

## Dataset

```
Database Systems and Logic Programming (DBLP)
Yelp Reviews (Yelp)
Foursquare
```

## Data description

You may need to prepare the data in the following format：

```bash
* data_dblp.pkl
#dict
data = {} 
#the initial emebdding of nodes, the key is node type.
data['feature'] = {'P':p_emb, 'A':a_emb,'V':v_emb} 
#Hierarchical tree structures(HTS), i.e., VPAc, APVC.
data['HTS']=[['V','P','A','C'],['A','P','V','C']]
#The adjacency matrix between each two levels in each hierarchical tree
data['adjs']=[[PV,AP],[PA,VP]]
#Data mapping for triples
data['data']

* ratings.txt
#This logs the items the user clicked on, where 1 is yes and 0 is no

* kg.txt
#Knowledge graph file with head entities in the first column, tail entities in the second column, and relations in the third column

* User_list.txt
#This is a file of users and their IDs.The first column contains the user's id, and the second column contains user ratings.txt, which records the items that the user clicked, with 1 if they clicked and 0 if they didn't

* Entity_list.txt
#This is a file of entities and their IDs
```
