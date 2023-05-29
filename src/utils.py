import bz2
import glob
import gzip
import os
from os.path import join

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
    ''' Concatenates txt files in a new vectors.txt file.
    
    Args:
        file_regex (str): the path to the files to combine, can contain wildcards.
        outpath (str): path to a file that the result is to be output in.
        
    Returns:
        none
        
    '''
    assert file_regex is not None, "No path to the files given, please specify."
    assert outpath is not None, "No path to output files given, please specify."

    output_folder = os.path.dirname(outpath)
    print
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
                         


def write_embedded_file(file_type: str = "txt", file_name: str = None, entities: list = None, embeddings: list = None):
    ''' Writes embeddings and entities in a file in a way that GEval can read.
    
    Args:
        file_type (str): the file type to be used. Standard is txt.
        file_name (str): the file name that should be used. Will be written to models folder.
        entities (list): the IRIs or names of the embedded entities.
        embeddings (list): the embeddings that should be written in the file.
        
    Returns:
        none
        
    '''

    assert file_name is not None, "No file name given, please specify."
    assert entities is not None, "No entities given, please give an array containing at least one."
    assert embeddings is not None, "No embedding vectors given, please give an array containing at least one."

    if(file_type == "txt"):
        file_type = ".txt"
    else: #only allow intended file types
        return

    with open(join("models", file_name + file_type), "w") as f:
        for t, e in zip(entities, embeddings):
            f.write(t)
            for v in e:
                f.write(" {}".format(v))
            f.write("\n")