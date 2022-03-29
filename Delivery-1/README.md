Delivery 1: Collocations
==================
In the first delivery of the term project, we will be working on extracting the collocations from the dataset.

How to execute the scripts?
------------------
We have developed the Python scripts on a Windows OS. Please execute the following on a bash terminal:

`./pipeline.sh`

The Shell script will execute all necessary Python scripts in order. That is, it will start from data cleaning, tokenization, n-gram extraction to collocation algorithms. Please feel free to change the pipeline and play with it.

Directory Structure
------------------
It is important to place the raw dataset inside a directory called `data`. For instance, if your dataset directory is called `2021-01`, then the directory structure should look like the following:

    Delivery-1
       |
       |--- data:        The directory which contains all dataset related files
       |   |
       |   |--- 2021-01:   A directory which contains all raw dataset files in JSON format
       |
       |--- <python scripts>
