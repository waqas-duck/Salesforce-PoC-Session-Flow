#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for working with embedding matrices.
"""

import numpy as np
import pandas as pd


class embeddings:
    def __init__(self, embeddings, rownames):
        self.embeddings = np.asarray(embeddings, dtype=float)
        if isinstance(rownames, (str, bytes)):
            self.rownames = [rownames]
        else:
            try:
                self.rownames = list(rownames)
            except TypeError:
                self.rownames = [rownames]

        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, -1)

        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 1D vector or 2D NumPy array")

        if len(self.rownames) != self.embeddings.shape[0]:
            raise ValueError("rownames length must match embeddings rows")
    ############################
    def shape(self):
        return self.embeddings.shape

    ############################
    def get_dataframe(self):
        dataframe = pd.DataFrame(self.embeddings.copy())
        dataframe.insert(0, "SESSION_ID", self.rownames)
        return dataframe

    ############################
    def select(self, names):
        if isinstance(names, (str, bytes)):
            names = [names]
        else:
            names = list(names)

        rownames = np.array(self.rownames)
        mask = ~np.isin(rownames, np.array(names))

        return self.__class__(
            embeddings=self.embeddings[mask, :],
            rownames=rownames[mask].tolist(),
        )

    ############################
    def get(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        rownames = np.array(self.rownames)
        mask = rownames == name

        if not mask.any():
            raise KeyError(f"name not found in rownames: {name}")

        return self.embeddings[mask, :][0]

    ############################
    def get_by_idx(self, idx):
        return self.embeddings[idx,:]

    ############################
    def calc_cosine_similarities(self, embedding, sort_by_similarity=False):
        embedding = np.asarray(embedding, dtype=float).ravel()

        if embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError("embedding length must match embeddings columns")

        dot_products = self.embeddings @ embedding
        embedding_norm = np.linalg.norm(embedding)
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        denominator = embedding_norms * embedding_norm

        similarities = np.full(self.embeddings.shape[0], np.nan)
        np.divide(dot_products, denominator, out=similarities, where=denominator != 0)

        similarities_df = pd.DataFrame({
            "rownames": self.rownames,
            "cosine_similarity": similarities,
        })

        if sort_by_similarity:
            similarities_df = similarities_df.sort_values(
                by="cosine_similarity",
                ascending=False,
            ).reset_index(drop=True)
                
        similarities_df = similarities_df.rename(columns={"cosine_similarity": "similarity"})
        return similarities_df
    ############################
    def calc_cosine_self_similarities(self, model):
        sims = model.similarity(self.embeddings, self.embeddings).numpy()
        #BUILD user_similarities
        sims = pd.DataFrame(sims, index=self.rownames, columns=self.rownames)
        sims.index.name = sims.columns.name= None        
        
        return sims
        
    ############################
