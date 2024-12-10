import torch

from itertools import count
from queue import PriorityQueue


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """
    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node):
        """ Adds a new beam search node to the queue of current nodes """
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        # Mark the node as finished
        node.finished = True
        # ensure all node paths have the same length for batch ops
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        # self.final.put((score, next(self._counter), node))

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        node = merged.get()
        node = (node[0], node[2])

        return node

    def prune(self):
        """
        Prunes the nodes by retaining only those that:
        - Are marked as finished.
        - Have a score greater than or equal to the highest score among finished nodes.

        This ensures the beam size remains constant while filtering out poor unfinished nodes.
        """
        nodes = PriorityQueue()

        # Find the best score among finished nodes
        best_finished_score = float('-inf')
        temp_nodes = []

        while not self.nodes.empty():
            score, _, node = self.nodes.get()
            temp_nodes.append((score, node))
            if node.finished:
                best_finished_score = max(best_finished_score, score)

        # Retain finished nodes or unfinished nodes above the best finished score
        pruned_nodes = [
            (score, node) for score, node in temp_nodes
            if node.finished or score >= best_finished_score
        ]

        # Sort by score and retain up to beam_size nodes
        pruned_nodes.sort(key=lambda x: x[0])  # Sort by score (ascending)
        pruned_nodes = pruned_nodes[:self.beam_size]

        # Rebuild the PriorityQueue with pruned nodes
        for score, node in pruned_nodes:
            nodes.put((score, next(self._counter), node))

        self.nodes = nodes


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):

        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search
        self.finished = False  # Add a flag to track if the hypothesis is completed

    def eval(self, alpha=0.0):
        """ Returns score of sequence up to this node 

        params: 
            :alpha float (default=0.0): hyperparameter for
            length normalization described in in
            https://arxiv.org/pdf/1609.08144.pdf (equation
            14 as lp), default setting of 0.0 has no effect
        
        """
        normalizer = (5 + self.length)**alpha / (5 + 1)**alpha
        return self.logp / normalizer
        