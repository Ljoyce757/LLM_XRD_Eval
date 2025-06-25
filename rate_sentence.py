from typing import List
from moverscore_v2 import word_mover_score
from collections import defaultdict
import numpy as np
def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score


refs = ['The dog bit the man.', 'The dog had bit the man.']
sys = 'The dog bit the man.'

a = moverscore = sentence_score(sys, refs)
print(a)