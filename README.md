# Distributed Link prediction in citation network
Used Spark with Scala.  Spark libraries : GraphX, Spark.ML<br /> 
In supervised approach, the features used for the training were according to publication Link prediction in citation networks, Naoki Shibata andYuya Kajikawa, Ichiro Sakata. <br />
In unsupervised approach only node informations are used.<br/>

## Data
training_set.txt : A set of 615,512 labeled node pairs (1 if there is an edge between two nodes, otherwise 0).<br /><br />
testing_set.txt : A set of 32,648 node pairs. The file contains one node per row, as: source node ID, target node ID.<br /> <br />
node_information.csv : For each paper out of 27,770 contains the following information :<br />
  <br/>
  1) unique ID<br/>
  2) publication year <br/>
  3) title <br/>
  4) authors <br/>
  5) name of journal(not available for all papers) <br/>
  6) abstract <br/>
  Abstracts are in lowercase.<br /><br />
Cit-Hepth.txt : The ground truth containing all the edges in our network.<br />


## Pre-processing

Using the graph constructed according to training data, some topological features are extracted to be used as input to features like difference in the number of in-links and the number of common neighbours between a pair of nodes.<br />
In addition, neighbours information are used to compute the link based jaccard coefficient. In-degree respresents the number "to" cited. Clustering was performed to see if two nodes belong to the same cluster.<br />
Finally, the features to  classifier  are : <br/> 
<br />
1)Difference in publication years<br/>
2)Number of common Authors<br/>
3)Self citation (if papers have at least one common author)<br/>
4)Same journal(if papers have been published to the same journal)<br/>
5)Same cluster(if papers belong to the same cluster)<br/>
6)Cosine similarity (taken from abstracts)<br/>
7)InDegrees difference (taken from graph)<br/>
8)Number of common neighbors (taken from graph)<br/>
9)Link based Jaccard  Coefficient( taken from graph)<br/>
10)Target’s in-degree (Number of times “to” cited)<br/>


## Classification Models in Supervised approach

Classifiers used : Logistic Regression, Decision Trees, Random Forests.<br/>
Best F1 Score achieved with Decision Trees, 0.95% <br/>


## Unsupervised approach

In order to avoid quadratic complexity, Local Sensitivity Hashing algorith MinHash is used. Experimentally, best results achieved for 15 hash tables in a single machine.<br/>
Input vector has 30.000 dimensions, 12.000 for title and 18.000 for abstract.<br/>
Best F1 Score was 10.4%.
