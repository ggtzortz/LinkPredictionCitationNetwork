# LinkPredictionCitationNetwork
Used Spark with Scala.  Spark libraries : GraphX, Spark.ML
 
In supervised approach, the features used for the training were according to publication Link prediction in citation networks, Naoki Shibata andYuya Kajikawa, Ichiro Sakata .
In unsupervised approach  

## Data
training_set.txt : A set of 615,512 labeled node pairs (1 if there is an edge between two nodes, otherwise 0).
testing_set.txt : A set of 32,648 node pairs. The file contains one node per row, as: source node ID, target node ID.
node_information.csv : For each paper out of 27,770 contains the following information :  1. unique ID, 2. publication year 3. title, 4. authors, 5. name of journal(not available for all papers) and 6. abstract. Abstracts are in lowercase.
Cit-Hepth.txt : The ground truth containing all the edges in our network.


## Pre-processing

Using the graph constructed according to training data, some topological features are extracted to be used as input to features like difference in the number of in-links and the number of common neighbours between a pair of nodes.
In addition, neighbours information are used to compute the link based jaccard coefficient. In-degree respresents the number "to" cited. Clustering was performed to see if two nodes belong to the same cluster.


### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

