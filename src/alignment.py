import os

import numpy as np
import pandas as pd
from numpy.linalg import svd


def absolute_orientation(target_vector_file: str = None, source_vector_file: str = None, outfile: str = None):
    """Calculates the absolute orientation following the paper of Sunipa Dev, Safia Hassan and Jeff Philips.
    Adaptation of the implementation of Jan Portisch et al.
    
    Args:
        target_vector_file (str): path to the vector file that should be used as the target of the rotation.
        source_vector_file (str): path to the vector file that should be rotated.
        outfile (str): path to location where rotated vector file should be stored.

    Returns:
        none

    """
    assert target_vector_file is not None, "No path for target vector file given, please specify one."
    assert source_vector_file is not None, "No path for source vector file given, please specify one."
    assert outfile is not None, "No path to output result given, please specify one."

    source_embedding = pd.read_csv(
                os.path.join(".", "models", source_vector_file),
                header=None,
                delim_whitespace=True,
            )
    source_embedding.set_index(0, inplace=True)

    target_embedding = pd.read_csv(
                os.path.join(".", "models", target_vector_file),
                header=None,
                delim_whitespace=True,
            )
    target_embedding.set_index(0, inplace=True)

    # filter source and target with them
    filtered_source_embeddings = source_embedding[source_embedding.index.isin(target_embedding.index)]
    filtered_source_embeddings = filtered_source_embeddings.to_numpy()

    filtered_target_embeddings = target_embedding[target_embedding.index.isin(source_embedding.index)]
    filtered_target_embeddings = filtered_target_embeddings.to_numpy()

    # get outer product
    outer_product = np.einsum(
        "ij,ik->jk", filtered_target_embeddings, filtered_source_embeddings
    )

    # get svd
    U, s, VT = svd(outer_product)
    rotation_matrix = U @ VT

    # rotate source
    source = source_embedding.to_numpy()
    rot_source = rotation_matrix @ source.T
    source_embedding = pd.DataFrame(
        rot_source.T,
        columns=source_embedding.columns,
        index=source_embedding.index
    )

    # write source into txt file
    source_embedding.to_csv(os.path.join(".", "models", outfile), header=False, sep=' ')