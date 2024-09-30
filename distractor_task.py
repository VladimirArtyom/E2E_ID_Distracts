import os
import json

from argparse import ArgumentParser, Namespace
from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from torch import Tensor
from typing import List


@register_task("distractor_task")
class DistractorTask(FairseqTask):
    
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("data_path",
                            metavar= "FILE", help="main dossier pour training")
        parser.add_argument("--max_positions",
                            default=1024,
                            type=int, help="max source/context length in a list")
        parser.add_argument("--max_tgt_positions",
                             default=10, type=int, help="max distractor length in a list" )
        parser.add_argument("--max_q_positions",default=10,
                            type=int, help="max question length in a list")
        parser.add_argument("--max_ans_positions", default=10,
                            type=int, help="max correct answer length in a list")
        parser.add_argument("--max_state_positions", default=10,
                            type=int)
    
    @classmethod
    def setup_task(cls, args: Namespace, **kwargs):
        input_vocab: Dictionary = Dictionary(os.path.join(args.data, "vocab_en.txt"))

        print(f"dictionary has : {len(input_vocab)} total of vocab")
        
        return cls(args, input_vocab)
        #return DistractorTask(args, input_vocab)

    def __init__(this, args: Namespace, input_vocab: Dictionary):
        super().__init__(args)
        this.args = args
        this.input_vocab = input_vocab

    def load_dataset(this, split: str,
                    combine: bool = False,
                    task_cfg: FairseqDataclass = None, **kwargs):
        f_path = os.path.join(this.args.data, "race_{}.json".format(split))

        contexts: List = []
        contexts_lengths: List = []

        distractors: List = []
        distractors_length: List = []

        questions: List = []
        questions_length: List = []

        #             return text[-max_len:]cut_end
        answers: List = []
        answers_length: List = []
        with open(f_path, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)

                context: List[str] = this.truncate_sequence(j["article"], this.args.max_positions - 1)
                question: List[str] = this.truncate_sequence(j["question"], this.args.max_q_positions - 1)
                distractor: List[str] = this.truncate_sequence(j["distractor"], this.args.max_tgt_positions - 1)
                answer: List[str] = this.truncate_sequence(j["answer_text"], this.args.ans_positions - 1)
                
                context_str: str =  " ".join(context)
                question_str: str  = " ".join(question)
                distractor_str: str  = " ".join(distractor)
                answer_str: str = " ".join(answer)

                add_if_not_e: bool = False

                tokens: Tensor = this.input_vocab.encode_line(
                    context_str,
                    add_if_not_exist=add_if_not_e).long()
                
                contexts.append(tokens)
                contexts_lengths.append(tokens.numel())

                tokens: Tensor = this.input_vocab.encode_line(
                    question_str,
                    add_if_not_exist=add_if_not_e
                ).long()

                questions.append(tokens)
                questions_length.append(tokens.numel())

                tokens: Tensor = this.input_vocab.encode_line(
                    distractor_str, 
                    add_if_not_exist=add_if_not_e
                ).long()

                distractors.append(tokens)
                distractors_length.append(tokens.numel())

                tokens: Tensor = this.input_vocab.encode_line(
                    answer_str,
                    add_if_not_exist=add_if_not_e
                ).long()
        assert len(contexts) == len(distractors)
        print("| {} {} {} examples".format(this.args.data, split, len(contexts)))

        this.datasets[split] = ...

    
    def truncate_sequence(this, text: str, max_len: int, cut_start: bool=True):
        if len(text) <= max_len:
            return text
        if cut_start:
            return text[-max_len:] # hello --> oll if cut_then end with max len 2
        return text[:max_len]
    
