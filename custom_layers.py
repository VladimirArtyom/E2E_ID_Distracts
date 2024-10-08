import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from torch.nn.modules import Adaptive

from fairseq.models import (FairseqEncoder, FairseqDecoder)
from helpers import initEmbedding, initLSTM, initLinear
from fairseq import utils
from fairseq.data import Dictionary
from typing import Tuple
from helpers import create_mask, len_to_mask
from fairseq.modules import AdaptiveSoftmax

import torch.utils

class DistractorEncoder(FairseqEncoder):
    def __init__(
        self, dictionary: Dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = initEmbedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = initLSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.lstm_second = initLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

        self.attention = Attention()
        self.dropout = nn.Dropout(dropout_in)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.output_units*2, self.output_units),
            nn.Tanh()
        )
        self.distance_layer = nn.Bilinear(self.output_units, self.output_units, 1)
        self.gating_layer = nn.Linear(self.output_units, 1)
        self.relu = nn.ReLU()
    def __init__(this, dictionary,
                 embed_dim: int=512, hidden_size: int=512, num_layers: int=1,
                 dropout_out: float=0.1, dropout_in: float=0.1, bidirectional: bool=False,
                 left_pad: bool=True, pretrained_embed: Tensor=None, padding_value: int=0):
        super().__init__(dictionary)
        this.num_layers = num_layers
        this.bidirectional = bidirectional
        this.hidden_size = hidden_size
        this.dropout_in = dropout_in
        this.dropout_out = dropout_out

        this.left_padding = left_pad
        this.padding_indx = dictionary.pad()
        this.padding_value = padding_value

        this.output_units = hidden_size
        if bidirectional:
            this.output_units = this.output_units * 2

        if pretrained_embed is not None:
            this.embed_tokens = pretrained_embed
        else:
            this.embed_tokens = nn.Embedding( num_embeddings=len(dictionary),
                                              embedding_dim=embed_dim,
                                              padding_idx=this.padding_indx)

        this.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=this.hidden_size,
            num_layers=this.num_layers,
            dropout=this.dropout_out if num_layers > 1 else 0,
            bidirectional=this.bidirectional)
        this.dropout = nn.Dropout(dropout_in)
        this.attention = Attention()
        this.gating_layer = nn.Linear(this.output_units, 1)
        this.distance_layer = nn.Bilinear(this.output_units, this.output_units, 1)
        this.fusion_layer = nn.Sequential( 
            nn.Linear(this.output_units*2, this.output_units),
            nn.Tanh()
        )
        this.relu = nn.ReLU()

    def encode_text(this, tokens: Tensor, lengths: Tensor, 
                     required_embed: bool = True, required_sort: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = tokens.size(0)
        sequence_len = tokens.size(-1)
        if required_embed:
            x: Tensor = this.embed_tokens(tokens)
            x = this.dropout(x)
        else: 
            x = tokens

        x_length_sorted, x_indices = torch.sort(lengths, descending=True)
        x_sorted = x.index_select(dim=0, index=x_indices)
        _, x_original_indices = torch.sort(x_indices)
        # Sorting based on the length of the sequence, Descending to make computation faster
        # x_packed is the packed version of x_sorted i.e the values are sorted by length
        x_packed: PackedSequence = nn.utils.rnn.pack_padded_sequence(x_sorted,
                                                     lengths=x_length_sorted.cpu(),
                                                     batch_first=True) #[Batch, Length, Embedding_dim]
        if this.bidirectional:
            state_size = ( 2* this.num_layers, batch_size, this.hidden_size)
        else:
            state_size = (this.num_layers, batch_size, this.hidden_size)

        h_0 = x.new_zeros(*state_size)
        c_0 = x.new_zeros(*state_size)

        x_packed_out, (final_hidden_state, final_cell_state) = this.lstm(x_packed, (h_0, c_0))
        #print(final_hidden_state.shape) # [num_layers, batch_size, hidden_size]
        #print(final_cell_state.shape) # [num_layers, batch_size, hidden_size]
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed_out,
                                                padding_value=this.padding_value, batch_first=True)
        x = this.dropout(x)
        #After processing sorted version of the sequence, Need to restore it before fed into another LAYER
        #All the x_packed_out will be sorted based on the original indices, same ase the hidden state and cell state
        x = x.index_select(dim=0, index=x_original_indices) # Get the original order of value not LENGTH, in here batch size
        final_hidden_state: Tensor = final_hidden_state.index_select(dim=1, index=x_original_indices) # Re-arange batch-size to original
        final_cell_state: Tensor = final_cell_state.index_select(dim=1, index=x_original_indices)

        if this.bidirectional:
            final_hidden_state = this.combine_bidirectional(final_hidden_state, batch_size)
            final_cell_state = this.combine_bidirectional(final_cell_state, batch_size)

        return x, final_hidden_state, final_cell_state # Return the input sequence, the calculation of hidden state and cell state
    
    def combine_bidirectional(this, outputs: Tensor, batch_size: int) -> Tensor:
        out: Tensor = outputs.view(this.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous() # [num_layers, batch_size, 2, hidden_size]
        return out.view(this.num_layers, batch_size, -1) #[num_layers, batch_size, hidden_size], It combined the result of bidirectional
    
    
    def forward(this, source_tokens: Tensor, source_lengths: Tensor,
                question_tokens: Tensor, question_lengths: Tensor,
                answer_tokens: Tensor, answer_lengths: Tensor):
        # Max length and ..._lengths are different
        # Max length represent the max length of the given sequence, most like padded into its maximum length
        # source lengths are the length of the source sequence [["i", "love", "you"], ["bs", "aa"]] -> [3, 2]
        batch_size, source_sequence_max_len = source_tokens.size()
        _, question_sequence_max_len = question_tokens.size()
        _, answer_sequence_max_len = answer_tokens.size()

        x_source, source_hidden_state, source_cell_state = this.encode_text(source_tokens, source_lengths)
        x_question, question_hidden_state, question_cell_state = this.encode_text(question_tokens, question_lengths, required_sort=True)
        x_answer, answer_hidden_state, answer_cell_state = this.encode_text(answer_tokens, answer_lengths, required_sort=True)
        print("this is fine")
        print(x_answer.shape)
        #x_source, x_question, x_answer --> [batch_size, sequence_length_max(each entity), hidden_size]
        ## Need mask for doing attention
        mask_attention_source_for_answer = create_mask(answer_lengths, source_lengths,
                                                answer_sequence_max_len, source_sequence_max_len)

        ## Find relevant information from the source for answer
        answer_attention, _ = this.attention(query=x_answer,
                                             key=x_source,
                                             value=x_source,
                                             mask=mask_attention_source_for_answer,
                                             dropout=this.dropout) #[batch_size, sequence_length_max, hidden_size ]
        # Combine the attention with the answer
        x_answer = this.fusion_layer(torch.cat([x_answer, answer_attention], dim=-1))

        mask_attention_source_for_question = create_mask(question_lengths, source_lengths,
                                                question_sequence_max_len, source_sequence_max_len)
        
        # Find relevant information from the source for question
        question_attention, _ = this.attention(query=x_question,
                                            key=x_source,
                                            value=x_source,
                                            mask=mask_attention_source_for_question,
                                            dropout=this.dropout)

        # Combine the attention with the question       
        x_question = this.fusion_layer(torch.cat([x_question, question_attention], dim=-1))

        answer_mask_for_self_attention = len_to_mask(answer_lengths, answer_sequence_max_len)
        # Basically we are doing self attention on the answer, well slightly. We'are trying to find which part
        # of the answer that is important. Just take a look at the calculation
        answer_self_attention = this.gate_self_attention(x_answer, mask=answer_mask_for_self_attention).unsqueeze(1) #[batch_size, sequence_length, hidden_size]

        ## Calculate the distance between the question and the answer itself using billinear
        ## This is useful because we need to know the distance between the question and the answer
        ## so that we can highlight the relevant questions given the fucking same answer
        question_answer_distance = this.distance_layer(x_question,
                                                       answer_self_attention.repeat(1, question_sequence_max_len, 1)) #[batch_size, sequence_length_max, 1]

        related_part_of_question_with_the_answer = x_question * question_answer_distance ## USE THIS TO HIGHLIGHT THE RELEVAN Question
     
        ## Calculate the question and answer to get the fucking distractors . i e related answers
        distractor_question_answer_mask = create_mask(answer_lengths,
                                                       question_lengths, answer_sequence_max_len,
                                                       question_sequence_max_len)

        distractor_qa_attention, _ =  this.attention(query=x_answer,
                                                     key=x_question,
                                                     value=x_question,
                                                     mask=distractor_question_answer_mask,
                                                     dropout=this.dropout)
        
        fusion_distractor_question_avec_answer = this.fusion_layer(torch.cat([x_answer, distractor_qa_attention], dim=-1))

        ## Now we need to find the probability of each answer using self attention , which answer to forget and which are not
        ## It basically related with the given question before
        distractor_self_attention = this.gate_self_attention(fusion_distractor_question_avec_answer,
                                                             mask=answer_mask_for_self_attention)
        distractor_self_attention = distractor_self_attention.unsqueeze(1)

        ## Calculate the distance between the distractor_Q_avec_answer with the source / context
        distractor_source_qa_distance = this.distance_layer( x_source.contiguous(),
            distractor_self_attention.repeat(1, source_sequence_max_len, 1))
        
        related_part_of_source_with_the_distractor_and_question = x_source * distractor_source_qa_distance

        ##### Finding relevant part within the context given a related question with the true answer ONLY AND 
        # not distractors

        relevant_part_mask = create_mask(source_lengths, question_lengths,
                                          source_sequence_max_len, question_sequence_max_len)

        relevant_attention, _ = this.attention(query=related_part_of_source_with_the_distractor_and_question,
                                                key=related_part_of_question_with_the_answer,
                                                value=related_part_of_question_with_the_answer,
                                                mask=relevant_part_mask,
                                                dropout=this.dropout)        
        # We are fusioning the combined attention from question, answer, distractors but not with information of related question_answer
        # with a new attention from the source that has been marked based on all relevant information of the question and answer ( related_part_of_question_with_the_answer)
        fusion_related_part_of_source_with_all_components = this.fusion_layer(torch.cat([related_part_of_source_with_the_distractor_and_question,
                                                                                         relevant_attention], dim=-1)) 


        question_final_outs, question_final_hidden, question_final_cell = this.encode_text(related_part_of_question_with_the_answer, question_lengths,
                                                                                            required_embed=False, required_sort=True)

        source_final_outs, source_final_hidden, source_final_cell = this.encode_text(fusion_related_part_of_source_with_all_components, source_lengths,
                                                                                            required_embed=False, required_sort=False)

        decoder_question_mask_for_self_attention = len_to_mask(question_lengths, question_sequence_max_len)
        #return question_final_outs, source_final_outs
        
        return {
            "encoder_out": (source_final_outs, source_final_hidden, source_final_cell,
                             question_final_outs, question_final_hidden, question_final_cell,),
            "encoder_padding_mask": decoder_question_mask_for_self_attention,
        }


    def gate_self_attention(this, sequences: Tensor, mask: Tensor=None):
        # Sequences - > [Batch_size, sequence_length, hidden_size]
        outs: Tensor = this.gating_layer(sequences).squeeze() #[batch_size, sequence_length] <-- value within sequence lenght has been changed

        if mask is not None:
            outs = outs.masked_fill(mask == 0, -65500)
        #print("outs",outs.shape)
        attention = F.softmax(outs, dim=-1) # [batch_size, sequence_length]
        attention = attention.unsqueeze(2) # [batch_size, sequence_length, 1]
        #print("attention",attention.shape)
        sequence_score = sequences * attention # [batch_size, sequence_length, 1]

        return sequence_score.sum(dim=1) # [batch_size, 1, hidden_size] i.e the score of each sequence

    def max_positions(this):
        return int(1e5)
        

class DistractorDecoder(FairseqDecoder):
    def __init__(this, dictionary,
                 embed_dim: int=512,
                 hidden_size: int=512,
                 out_embed_dim: int = 512,
                 num_layers: int=1,
                 dropout_in: float=0.1,
                 dropout_out: float=0.1,
                 attention: bool = True,
                 encoder_output_units=512,
                 pretrained_embed = None,
                 share_input_output_embed=False,
                 adaptive_softmax_cutoff=None,
                 proj_initial_state=False):
        super().__init__(dictionary=dictionary)

        this.hidden_size = hidden_size
        this.embed_dim = embed_dim
        this.share_input_output_embedding = share_input_output_embed

        this.dropout_in = dropout_in
        this.dropout_out = dropout_out

        this.adaptive_softmax = None
        this.num_embeddings = len(dictionary)
        this.need_attention = True
        this.padding_indx = dictionary.pad()

        this.encoder_output_units = encoder_output_units
        if this.encoder_output_units != hidden_size:
            this.encoder_hidden_projector = initLinear(encoder_output_units, hidden_size)
            this.encoder_cell_projector = initLinear(encoder_output_units, hidden_size)
        else:
            this.encoder_hidden_projector = None
            this.encoder_cell_projector = None


        if pretrained_embed is None:
            this.embed_tokens = initEmbedding(this.num_embeddings, this.embed_dim,
                                              this.padding_indx)
        else:
            this.embed_tokens = pretrained_embed

        if attention:
            this.attention = AttentionLayer(this.hidden_size,
                                            this.encoder_output_units,
                                            this.hidden_size)
        else:
            this.attention = None

        if adaptive_softmax_cutoff is not None:
            this.adaptive_softmax = AdaptiveSoftmax(this.num_embeddings,
                                                    this.hidden_size,
                                                    adaptive_softmax_cutoff, 
                                                    dropout=dropout_out)
        elif this.share_input_output_embedding == False:
            this.fc_out = initLinear(out_embed_dim, this.num_embeddings)
        
        this.match_layer = nn.Bilinear(hidden_size, hidden_size, 1)
        this.gate_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        this.relu =  nn.ReLU()

        this.project_initial_state = proj_initial_state
        if this.project_initial_state:
            this.project_hiddens = initLinear(encoder_output_units, encoder_output_units)
            this.project_cells = initLinear(encoder_output_units, encoder_output_units)
    def forward(this, prev_output_tokens: Tuple, encoder_out: Tuple, incremental_state=None):
        ...


class AttentionLayer(nn.Module):
    def __init__(this, input_embedding_dim: Tensor,
                 source_embedding_dim: Tensor,
                 output_embedding_dim: Tensor,
                 bias=False):
        this.input_projector = initLinear(input_embedding_dim, source_embedding_dim, bias=bias)
        this.output_projector = initLinear(input_embedding_dim + source_embedding_dim, output_embedding_dim, bias=bias)

    def forward(this, input: Tensor,
                 source_hidden: Tensor,
                 encoder_padding_mask: Tensor):
        #input [batch_size, source_embedding_dim]
        #source_hidden [src_len, batch_size, output_embed_dim]
        #encoder_pad_mask [batch_size, src_len, hidden_size]
        
        #x1 [batch_size, source_embedding_dim]
        x1: Tensor = this.input_projector(input)

        attention_score = (source_hidden * x1.unsqueeze(0)).sum(dim=2) #-->[src_len, batch_size] #the represent the attention score for each sentence
        if encoder_padding_mask is not None:
            attention_score = attention_score.float().masked_fill_(encoder_padding_mask, 7e-5).type_as(attention_score)
        
        attention_score = F.softmax(attention_score, dim=0) # Porque src_len,batch_size consist the valeur of the attention, softmax will get the probability for each sentence
        # The probability is related with the source or the given context

        ## As the attention score now is the size of [src_len, batch_size] each of the observation consist the 
        ## probability for each sentence given the context.
        
        ## Now we would like to element-wise this probability with their corresponding source hidden
        ## attention_score[src_len, batch_size, 1] --> [src_len, batch_size, hidden_size]
        ## Result --> [batch_size, hidden_size]

        x1 = (attention_score.unsqueeze(dim=2) * source_hidden).sum(dim=0)
        output = torch.tanh(this.output_projector(torch.cat([x1, input], dim=1)))
        return output, attention_score

class Attention(nn.Module):

    def forward(this, query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Tensor = None,
                highlights: Tensor= None,
                dropout: nn.Dropout = None):

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.as_tensor(query.size(-1), dtype=torch.int64)) #[batch, num_heads, num_queries, num_keys]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -65500)
        
        attention = F.softmax(scores, dim=-1)
        
        if highlights is not None:
            assert attention.size(0) == highlights.size(0)
            assert attention.size(-1) == highlights.size(-1)
            attention = attention * highlights

            if mask is not None:
                attention = attention.masked_fill(mask==0, 7e-5)

            sum = attention.sum(-1, keepdim=True)
            attention = attention / sum

        if dropout is not None:
            attention = dropout(attention)
        
        return torch.matmul(attention, value), attention