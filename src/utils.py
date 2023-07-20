import bz2
import glob
import gzip
import os
from os.path import dirname, join, basename

import evaluation_framework as geval
from evaluation_framework.txt_dataManager import DocumentSimilarityDataManager, SemanticAnalogiesDataManager
import pandas as pd
import requests
import tqdm


def get_all_data(datasets: list = None, path: str = "data"):
    """Downloads all data and decompress them.
    
    Args:
        datasets (list): list of datasets to get.
        path (str): path to location where files should be stored.

    Returns:
        none

    """
    assert datasets is not None, "No datasets selected, please spefify at least one."
    assert path is not None, "No path given, please specify one."

    if not os.path.exists(path):
        os.makedirs(path)

    for link in datasets:
        file_name = link.rsplit("/", 1)[-1]
        zip_path = os.path.join(path, file_name)
        file_path = zip_path.rsplit(".", 1)[0]

        if not os.path.exists(zip_path) and not os.path.exists(file_path):
            download(link, zip_path)

        if not os.path.exists(file_path):
            decompress(zip_path)


def download(url: str = None, path: str = None):
    """Downloads files from a link to a path
    
    Args:
        url (str): URL where files can be found.
        path (str): path to location where files should be stored.

    Returns:
        none

    """
    assert url is not None, "No url given, please spefify one."
    assert path is not None, "No path given, please specify one."

    response = requests.get(url, stream=True)
    total_bytesize = int(response.headers.get('content-length', 0))
    progress_bar = tqdm.tqdm(total=total_bytesize, unit='iB', unit_scale=True)

    with open(path, "wb") as file:
        for data in response.iter_content(8192):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_bytesize != 0 and progress_bar.n != total_bytesize:
        raise FileExistsError


def decompress(path: str = None):
    ''' Decompresses files.
    
    Args:
        path (str): path to file that is to be decompressed.
        
    Returns:
        none
        
    '''
    assert path is not None, "No path given, please specify."

    if path.endswith("bz2"):
        new_path = os.path.splitext(path)[0]
        print("Decompressing:", new_path)
        with open(new_path, "wb") as new_file, bz2.BZ2File(path, "rb") as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)

        os.remove(path) # remove archive file to save space
    elif(path.endswith("gz")):
        new_path = os.path.splitext(path)[0]
        print("Decompressing:", new_path)
        with open(new_path, "wb") as new_file, gzip.open(path, "rb") as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)

    else:
        print("Unexpected filetype ecountered. gz- or bz2-file was expected.")


def combine_vectors(file_regex: str = None, outpath: str = None, duplicates: bool = True):
    ''' Concatenates txt files in a folder in a new vectors.txt file.
    
    Args:
        file_regex (str): the path to the files to combine, can contain wildcards.
        outpath (str): path to a file that the result is to be output in.
        duplicates (bool): should duplicates be kept?
        
    Returns:
        none
        
    '''
    assert file_regex is not None, "No path to the files given, please specify."
    assert outpath is not None, "No path to output files given, please specify."

    output_folder = os.path.dirname(outpath)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    entities = set()
    
    with open(outpath, "w") as outfile:
        for file in glob.glob(file_regex):
            with open(file, "r") as infile:
                for line in infile:
                    if duplicates:
                        outfile.write(line)
                    else:
                        e = line.split(' ', 1)[0]
                        if(e not in entities):
                            entities.add(e)
                            outfile.write(line)
                         

def combine_aligned_vectors(aligned_regex: str = None, target_file: str = None, outpath: str = None, duplicates: bool = True):
    ''' Concatenates aligned txt files in a folder and their target file in a new vectors.txt file.
    
    Args:
        aligned_regex (str): the path to the files to combine, can contain wildcards.
        target_file (str): the path to the target file. Will always be put in first place of the new file.
        outpath (str): path to a file that the result is to be output in.
        duplicates (bool): should duplicates be kept?
        
    Returns:
        none
        
    '''
    assert aligned_regex is not None, "No path to the files given, please specify."
    assert target_file is not None, "No path to the target file given, please specify."
    assert outpath is not None, "No path to output files given, please specify."

    output_folder = os.path.dirname(outpath)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    entities = set()
    files = glob.glob(aligned_regex)
    files.insert(0, glob.glob(target_file)[0])
    
    with open(outpath, "w") as outfile:
        for file in files:
            with open(file, "r") as infile:
                for line in infile:
                    if duplicates:
                        outfile.write(line)
                    else:
                        e = line.split(' ', 1)[0]
                        if(e not in entities):
                            entities.add(e)
                            outfile.write(line)
                         

def get_geval_entities(tasktype: str = None) -> set:
    ''' Makes a set of entities contained in the tasks of the evaluation framework GEval.
    
    Args:
        tasktype (str): "all" or task type name indicating which entities to get.
    Returns:
        set: the entities.
        
    '''
    tasktype_list = ["all", "Classification", "Clustering", "Regression", "DocumentSimilarity","EntityRelatedness", "SemanticAnalogies"]
    assert tasktype is not None, "No selection of task type given, please specify."
    assert tasktype in tasktype_list, ("Lease provide a task type from the following choices: all, Classification, Clustering, "
                                           "Regression, DocumentSimilarity, EntityRelatedness, SemanticAnalogies")

    geval_path = dirname(geval.__file__)
    filtered_entities = set()

    # Classification
    if(tasktype == "all" or tasktype == "Classification"):
        for file in glob.glob(join(geval_path, "Classification", "data", "*.tsv")):
            df = pd.read_csv(file, usecols=["DBpedia_URI15"], delim_whitespace=True)
            filtered_entities.update(set(df.iloc[:, 0]))

    # Clustering
    if(tasktype == "all" or tasktype == "Clustering"):
        for file in glob.glob(join(geval_path, "Clustering", "data", "*.tsv")):
            df = pd.read_csv(file, usecols=["DBpedia_URI"], delim_whitespace=True)
            filtered_entities.update(set(df.iloc[:, 0]))

    # Regression
    if(tasktype == "all" or tasktype == "Regression"):
        for file in glob.glob(join(geval_path, "Regression", "data", "*.tsv")):
            df = pd.read_csv(file, usecols=["DBpedia_URI15"], delim_whitespace=True)
            filtered_entities.update(set(df.iloc[:, 0]))

    # DocumentSimilarity
    if(tasktype == "all" or tasktype == "DocumentSimilarity"):
        dsm = DocumentSimilarityDataManager(debugging_mode=False)
        filtered_entities.update(set(dsm.get_entities(filename=join(geval_path, "DocumentSimilarity", "data", "LP50_entities.json")).iloc[:, 1]))

    # EntityRelatedness
    if(tasktype == "all" or tasktype == "EntityRelatedness"):
        with open(join(geval_path, "EntityRelatedness", "data", "KORE.txt"), "r") as file:
            for line in file:
                filtered_entities.add(line.strip())

    # SemanticAnalogies
    if(tasktype == "all" or tasktype == "SemanticAnalogies"):
        for file in glob.glob(join(geval_path, "SemanticAnalogies", "data", "*.txt")):
            with open(file, "r") as f:
                for line in f:
                    filtered_entities.update(set(line.rstrip().split()))

    return filtered_entities


def filter_vector_file(vector_file: str = None, outpath: str = None):
    ''' Filters the entities out of the vector file that are not used by GEval.
    
    Args:
        vector_file (str): the vector file to be filtered.
        outpath (str): the new vector file name that should be used.
    Returns:
        none
        
    '''
    assert vector_file is not None, "No vector file path given, please specify."
    assert outpath is not None, "No output file given, please specify."

    filtered_entities = get_geval_entities("all")

    output_folder = os.path.dirname(outpath)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # filter and write new vector file
    with open(outpath, "w") as outfile:
        with open(vector_file, "r") as infile:
            for line in infile:
                e = line.split(' ', 1)[0]
                if e in filtered_entities:
                    outfile.write(line)


def get_entity_distribution(partition_paths: list = None, tasktype: str = None) -> pd.DataFrame:
    ''' Calculates the distribution of entities used for evaluation in GEval in a given partition.
    
    Args:
        partition_path (list): list of paths to the files that hold the partitions.
        tasktype (str): "all" or task type that the distribution is to be done for.
    Returns:
        pd.DataFrame: collection of the entity distribution sorted by count.
        
    '''
    assert partition_paths is not None, "No list of paths given, please specify."
    assert tasktype is not None, "No selection of task type given, please specify."

    df = pd.DataFrame(columns=["file", "count", "share"])
    entities = get_geval_entities(tasktype)

    for path in partition_paths:
        partition = pd.read_csv(path, header=None, delim_whitespace=True, usecols=[0])
        count = partition.iloc[:, 0].isin(entities).sum()
        df = df.append({"file": path, "count": count, "share": (count / len(entities))}, ignore_index=True)

    return df
