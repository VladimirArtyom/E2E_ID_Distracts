
import numpy as np
import torch
from torch import Tensor
from fairseq.data import data_utils, FairseqDataset, Dictionary
from typing import List, Mapping

def collatttttteee( samples,
                    pad_idx: int,
                    eos_idx: int):
    ...

class DistractorDataset(FairseqDataset):
    def __init__(this,
                 contexts: List[Tensor],
                 contexts_lengths: List[int],
                 dictionary: Dictionary,
                 questions: List[Tensor],
                 questions_lengths: List[int],
                 distractors: List[Tensor],
                 distractors_lengths: List[int],
                 answers: List[Tensor],
                 answers_lengths: List[int],
                 left_pad_context: bool = True,
                 left_pad_distractors = False,
                 max_context_positions: int = 1024,
                 max_distractor_positions: int = 10,
                 max_question_positions: int = 10,
                
                shuffle: bool = True,
                input_feeding: bool = True,
                remove_eos_from_context: bool = False ,
                append_eos_to_target: bool =False,
                 ):
        this.contexts = contexts
        this.questions = questions
        this.distractors = distractors
        this.answers = answers

        this.contexts_lengths = np.array(contexts_lengths)
        this.questions_lengths = np.array(questions_lengths)
        this.distractors_lengths = np.array(distractors_lengths)
        this.answers_lengths = np.array(answers_lengths)

        this.dictionary = dictionary

        this.left_pad_context = left_pad_context
        this.left_pad_distractors = left_pad_distractors

        this.max_context_positions = max_context_positions
        this.max_target_positions = max_distractor_positions
        this.max_question_positions = max_question_positions

        this.shuffle = shuffle
        this.input_feeding = input_feeding
        this.remove_eos_from_context = remove_eos_from_context
        this.append_eos_to_target = append_eos_to_target


    def __getitem__(this, index: int) -> Mapping[str, str]:
        distractor = this.distractors[index]
        context = this.contexts[index]
        question = this.questions[index]
        answer = this.answers[index]
        
        if this.append_eos_to_target:
            eos = this.dictionary.eos()
            if distractor[-1] != eos:
                distractor = torch.cat([distractor, torch.LongTensor([eos])])
        
        if this.remove_eos_from_context:
            eos = this.dictionary.eos()
            if context[-1] == eos:
                context = this.contexts[index][:-1]

        return {
            'id': index,
            'context': context,
            'distractor': distractor,
            'question': question,
            'answer': answer
        }

    def __len__(this):
        return len(this.contexts)
    
    def collater(this, samples):
        ...

    def num_tokens(this, index):
        ...
    
    def max_positions(this):
        return this.args.max_positions, this.args.max_tgt_positions
    def num_tokens_vec(this, indices):
        ...

    def size(this, index):
        ...

    def prefetch(this, indices):

    