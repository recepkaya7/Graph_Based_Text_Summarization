import json
import string
from nltk.stem.porter import *

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import BertTokenizer

# import torch, transformers, tokenizers
# torch.__version__, transformers.__version__, tokenizers.__version__

import nltk

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QFileDialog

import networkx as nx
import matplotlib.pyplot as plt

import random

import yazlab3



class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        
        self.al = yazlab3

        self.setGeometry(300, 300, 600, 600)
        self.setWindowTitle('Dosya Yükleme Uygulaması')



        self.btn = QPushButton('Graf göster',self)
        self.btn.move(20,20)
        
        self.btn.clicked.connect(yazlab3.draw_graph)



        self.show()

    

    

class Graph:
    def __init__(self,sentences,scoree,con,baglanti):
        self.sentences = sentences
        self.scoree = scoree
        self.con = con
        self.baglanti = baglanti

    def show_graph(self):
        # Create a new graph
        G = nx.Graph()
        print(self.baglanti)
        # Add nodes for each sentence

        # Add nodes for each sentence
        for i, sentence in enumerate(self.sentences):
            G.add_node(i+1, label=sentence)

        # Add edges between sentences in the same paragraph
        for i in range(len(self.sentences)-1):
            for j in range(i+1, len(self.sentences)):
                    print(i,j)
                    G.add_edge(i+1, j+1,weight = self.baglanti[i][0][j])

        # Set node labels and scores
        labels = nx.get_node_attributes(G, 'label')

       
        
        scores = self.scoree
        nx.set_node_attributes(G, scores, 'score')

        connection_counts = self.con
        print(connection_counts)
        
        nx.set_node_attributes(G, connection_counts, 'connection_counts')

        # Draw graph with node labels, scores, and connection counts
        pos = nx.spring_layout(G, seed=3234, k=10)
        nx.draw(G, pos, labels=labels, with_labels=True, font_size=10)
        node_scores = nx.get_node_attributes(G, 'score')
        node_connection_counts = nx.get_node_attributes(G, 'connection_counts')
        for node, score in node_scores.items():
            
            plt.text(pos[node][0], pos[node][1] + 0.05, f'cümle skoru={score[node-1]}', horizontalalignment='center', fontsize=10)
            plt.text(pos[node][0], pos[node][1] - 0.1, f'{connection_counts[node-1]}', horizontalalignment='center', fontsize=10)

        edge_weights = nx.get_edge_attributes(G, 'weight')
        for (u, v, weight) in G.edges(data='weight'):
            plt.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, f'{weight}', horizontalalignment='center', fontsize=10)

        
           
        plt.show()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    #a = SentenceList()
    #a.getir()
    sys.exit(app.exec_())


