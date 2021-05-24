import sys
import copy
sys.path.append(".")
import os
from collections import defaultdict
import argparse
import json
import string
import random
import numpy as np
from datetime import datetime
import torch
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, AutoModel, AutoConfig, AutoModelWithLMHead
from scripts.triviaqa_utils import evaluation_utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDataParallel

from longformer.longformer import Longformer
from longformer.sliding_chunks import pad_to_window_size

from pytorch_lightning.loggers import WandbLogger

import hiddenlayer as hl
import wandb
from scripts.triviaqa_new_pl import TriviaQADataset, TriviaQA

class ModifiedTriviaQADataset(TriviaQADataset):
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len,
                 num_of_interactions):
        super().__init__(file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len)
        self.num_of_interactions = num_of_interactions


    def __getitem__(self, idx):
        entry = self.data_json[idx]
        return self.one_example_to_tensors(entry, idx)


    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        paragraph = example["paragraphs"][0]
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        qa = paragraph["qas"][0]
        original_question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        answer_spans = []
        question_paraphrases = qa["question_paraphrases"][:self.num_of_interactions]
        question_paraphrases.insert(0, original_question_text)
        tensors_per_paraphrase_list = []
        for answer in qa["answers"]:
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            try:
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                token_ids = self.tokenizer.encode(orig_answer_text)
            except RuntimeError:
                print(f'Reading example {idx} failed')
                start_position = 0
                end_position = 0
            answer_spans.append({'start': start_position, 'end': end_position,
                                 'text': orig_answer_text, 'token_ids': token_ids})

        # ===== Given an example, convert it into tensors  =============

        # these lists are per paraphrased question

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        start_positions_list = []
        end_positions_list = []
        answer_token_ids_list = []

        for question_idx, question_text in enumerate(question_paraphrases):
            query_tokens = self.tokenizer.tokenize(question_text)
            query_tokens = query_tokens[:self.max_question_len]
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                # hack: the line below should have been `self.tokenizer.tokenize(token')`
                # but roberta tokenizer uses a different subword if the token is the beginning of the string
                # or in the middle. So for all tokens other than the first, simulate that it is not the first
                # token by prepending a period before tokenizing, then dropping the period afterwards
                sub_tokens = self.tokenizer.tokenize(f'. {token}')[1:] if i > 0 else self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            all_doc_tokens = all_doc_tokens[:self.max_doc_len]

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3
            assert max_tokens_per_doc_slice > 0
            slice_start = 0
            slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

            doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
            tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                                                + doc_slice_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 1)
            assert len(segment_ids) == len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            if question_idx == 0: # calculating start/end/answer only for original question
                    doc_offset = len(query_tokens) + 2 - slice_start
                    start_positions = []
                    end_positions = []
                    answer_token_ids = []
                    for answer_span in answer_spans:
                        start_position = answer_span['start']
                        end_position = answer_span['end']
                        tok_start_position_in_doc = orig_to_tok_index[start_position]
                        not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                        tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
                        if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                            # this answer is outside the current slice
                            continue
                        start_positions.append(tok_start_position_in_doc + doc_offset)
                        end_positions.append(tok_end_position_in_doc + doc_offset)
                        answer_token_ids.append(answer_span['token_ids'])
                    assert len(start_positions) == len(end_positions)
                    if self.ignore_seq_with_no_answers and len(start_positions) == 0:
                        continue

                    # answers from start_positions and end_positions if > self.max_num_answers
                    start_positions = start_positions[:self.max_num_answers]
                    end_positions = end_positions[:self.max_num_answers]
                    answer_token_ids = answer_token_ids[:self.max_num_answers]

                    # -1 padding up to self.max_num_answers
                    padding_len = self.max_num_answers - len(start_positions)
                    start_positions.extend([-1] * padding_len)
                    end_positions.extend([-1] * padding_len)
                    answer_token_ids.extend([[]] * padding_len)

                    # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
                    found_start_positions = set()
                    found_end_positions = set()
                    found_answer_token_ids = set()
                    for i, (start_position, end_position, answer_tokens) in enumerate(
                            zip(start_positions, end_positions, answer_token_ids)
                            ):
                        if start_position in found_start_positions:
                            start_positions[i] = -1
                        if end_position in found_end_positions:
                            end_positions[i] = -1
                        answer_tokens_as_str = ','.join([str(x) for x in answer_tokens])
                        if answer_tokens_as_str in found_answer_token_ids:
                            answer_token_ids[i] = []
                        found_start_positions.add(start_position)
                        found_end_positions.add(end_position)
                        found_answer_token_ids.add(answer_tokens_as_str)

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            if question_idx == 0:
                start_positions_list.append(start_positions)
                end_positions_list.append(end_positions)
                answer_token_ids_list.append(answer_token_ids)

        # pad answers in answer_token_ids_list to the longest answer
        max_answer_len = max([len(item) for sublist in answer_token_ids_list for item in sublist])  # flat list
        if max_answer_len == 0:
            max_answer_len = 2
        for answers_of_one_slice in answer_token_ids_list:
            for answer_tokens in answers_of_one_slice:
                if len(answer_tokens) == 0:
                    # TODO: <s></s><pad><pad><pad> or <pad><pad><pad><pad><pad> ?
                    padding_len = max_answer_len - len(answer_tokens) - 2
                    answer_tokens.extend([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id] +
                                         ([self.tokenizer.pad_token_id] * padding_len))
                else:
                    padding_len = max_answer_len - len(answer_tokens)
                    answer_tokens.extend([self.tokenizer.pad_token_id] * padding_len)

        return (torch.tensor(input_ids_list).unsqueeze(0), torch.tensor(input_mask_list).unsqueeze(0),
                             torch.tensor(segment_ids_list).unsqueeze(0),
                # NOTE: the labels provided are for the original question only.
                             torch.tensor(start_positions_list), torch.tensor(end_positions_list),
                             torch.tensor(answer_token_ids_list),
                             self._get_qid(qa['id']),  qa["aliases"])  # for eval


class InteractiveTriviaQA(TriviaQA):
    def __init__(self, args, current_interaction_num, max_num_of_interactions):
        super().__init__(args)
        self.current_interaction_num = current_interaction_num
        self.max_num_of_interactions = max_num_of_interactions
        self.learned_weighted_sum = torch.nn.Linear(self.max_num_of_interactions+1, 1)
        dataset = ModifiedTriviaQADataset(file_path=self.args.train_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=self.args.ignore_seq_with_no_answers,
                                  num_of_interactions=self.current_interaction_num)
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids, _, _ = dataset[0]
        self.example_input_array = (input_ids.to(self.device), input_mask.to(self.device), segment_ids.to(self.device),
                                   subword_starts.to(self.device), subword_ends.to(self.device), answer_token_ids.to(self.device))


    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions, answer_token_ids):

        batch_size = input_ids.shape[0]
        input_ids = input_ids.view(batch_size * (self.current_interaction_num+1), -1)
        attention_mask = attention_mask.view(batch_size * (self.current_interaction_num+1), -1)
        question_end_index = self._get_question_end_index(input_ids)
        # Each batch is one document, and each row of the batch is a chunck of the document.
        # Make sure all rows have the same question length.
        # assert (question_end_index[0].float() == question_end_index.float().mean()).item()
        # local attention everywhere, global attention on question
        tri = torch.tril(torch.ones([input_ids.shape[1],input_ids.shape[1]], dtype=torch.long, device=input_ids.device), diagonal=-1)
        attention_mask = tri[question_end_index] + 1

        # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)
        sequence_output = self.model.forward(input_ids, attention_mask=attention_mask)[0]
        sequence_output = sequence_output.view(batch_size, self.current_interaction_num+1, sequence_output.shape[1], -1)
        p = (0, 0, 0, 0, 0, self.max_num_of_interactions-self.current_interaction_num)
        sequence_output = torch.nn.functional.pad(sequence_output, p).permute(0,2,3,1)
        weighted_sum = self.learned_weighted_sum(sequence_output)
        weighted_sum.squeeze_(-1)
        logits = self.qa_outputs(weighted_sum)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # NOTE: this model predicts start and end index in the *original* question + context encoding.
            if not self.args.regular_softmax_loss:
                # loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf
                # NOTE: this returns sum of losses, not mean, so loss won't be normalized across different batch sizes
                # but batch size is always 1, so this is not a problem
                start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
                end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                start_positions = start_positions[:, 0:1]
                end_positions = end_positions[:, 0:1]
                start_loss = loss_fct(start_logits, start_positions[:, 0])
                end_loss = loss_fct(end_logits, end_positions[:, 0])

            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def validation_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids)
        if self.args.seq2seq:
            logit_scores = output[1]
            answer_score_indices = logit_scores.sort().indices
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=input_mask, use_cache=True,)
            answer_text = ''
            best_answer_score = 0
            for i in answer_score_indices:
                generated_answer_ids = generated_ids[answer_score_indices[i]]
                generated_answer_ids[-1] = self.tokenizer.eos_token_id
                index_of_eos_token = (generated_answer_ids == self.tokenizer.eos_token_id).nonzero()[0, 0].item()
                generated_answer_ids = generated_answer_ids[1:index_of_eos_token]
                answer_text = self.tokenizer.decode(generated_answer_ids)
                if answer_text != '':
                    best_answer_score = logit_scores[answer_score_indices[i]]
                    break
            f1_score = evaluation_utils.metric_max_over_ground_truths(evaluation_utils.f1_score, answer_text, aliases)
            em_score = evaluation_utils.metric_max_over_ground_truths(evaluation_utils.exact_match_score, answer_text, aliases)
            return {'vloss': output[0], 'vem': generated_answer_ids.new_zeros([1]).float(),
                    'qids': [qids], 'answer_scores': [best_answer_score],
                    'f1': [f1_score], 'em': [em_score]}


        loss, start_logits, end_logits = output[:3]
        answers = self.decode(input_ids[:,0,:], start_logits, end_logits)

        # each batch is one document
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)[0:1]
        qids = [qids]
        aliases = [aliases]

        f1_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.f1_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]
        # TODO: if slow, skip em_scores, and use (f1_score == 1.0) instead
        em_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.exact_match_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]
        answer_scores = [answer['score'] for answer in answers]  # start_logit + end_logit
        assert len(answer_scores) == len(f1_scores) == len(em_scores) == len(qids) == len(aliases) == 1

        # TODO: delete this metric
        pred_subword_starts = start_logits.argmax(dim=1)
        pred_subword_ends = end_logits.argmax(dim=1)
        exact_match = (subword_ends[:, 0].squeeze(dim=-1) == pred_subword_ends).float() *  \
                      (subword_starts[:, 0].squeeze(dim=-1) == pred_subword_starts).float()

        return {'vloss': loss, 'vem': exact_match.mean(),
                'qids': qids, 'answer_scores': answer_scores,
                'f1': f1_scores, 'em': em_scores}

    def test_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids)
        if self.args.seq2seq:
            raise NotImplemented

        loss, start_logits, end_logits = output[:3]
        answers = self.decode(input_ids[:,0,:], start_logits, end_logits)

        # each batch is one document
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)[0:1]
        qids = [qids]
        assert len(answers) == len(qids)
        return {'qids': qids, 'answers': answers}

    def training_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': input_ids.numel(),
                            'mem': torch.cuda.memory_allocated(input_ids.device) / 1024 ** 3}
        return {'loss': loss, 'log': tensorboard_logs}



    def train_dataloader(self):
        if self.train_dataloader_object is not None:
            return self.train_dataloader_object
        dataset = ModifiedTriviaQADataset(file_path=self.args.train_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=self.args.ignore_seq_with_no_answers,
                                  num_of_interactions=self.current_interaction_num)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=(sampler is None),
                        num_workers=self.args.num_workers, sampler=sampler,
                        collate_fn=ModifiedTriviaQADataset.collate_one_doc_and_lists)
        self.train_dataloader_object = dl
        return self.train_dataloader_object


    def test_dataloader(self):
        if self.test_dataloader_object is not None:
            return self.test_dataloader_object
        dataset = ModifiedTriviaQADataset(file_path=self.args.dev_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                          ignore_seq_with_no_answers=False, num_of_interactions=self.current_interaction_num)  # evaluation data should keep all example

        dl = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=self.args.num_workers, sampler=None,
                        collate_fn=ModifiedTriviaQADataset.collate_one_doc_and_lists)
        self.test_dataloader_object = dl
        return self.test_dataloader_object


    def val_dataloader(self):
        if self.val_dataloader_object is not None:
            return self.val_dataloader_object
        dataset = ModifiedTriviaQADataset(file_path=self.args.dev_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                          ignore_seq_with_no_answers=False, num_of_interactions=self.current_interaction_num)
        # evaluation data should keep all examples
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=self.args.num_workers, sampler=sampler,
                        collate_fn=ModifiedTriviaQADataset.collate_one_doc_and_lists)
        self.val_dataloader_object = dl
        return self.val_dataloader_object


    @staticmethod
    def add_interactive_specific_args(parser):
        parser.add_argument("--total_interactions_num", type=int, help="Total number of interactions available in this run")
        parser.add_argument("--current_added_interactions", type=int, help="Number of interactions added")
        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = InteractiveTriviaQA(args, current_interaction_num=args.current_added_interactions,
                                max_num_of_interactions=args.total_interactions_num)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        # log_graph=True
        version=0 # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename="checkpoints",
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        # save_last=True,
        mode='min',
        period=-1,
    )

    wandb_logger = WandbLogger(name=args.run_name, project=args.project_name)

    print(args)
    train_set_size = 110648  # hardcode dataset size. Needed to compute number of steps for the lr scheduler
    args.steps = args.epochs * train_set_size / (args.batch_size * max(args.gpus, 1))
    print(f'>>>>>>> #steps: {args.steps}, #epochs: {args.epochs}, batch_size: {args.batch_size * args.gpus} <<<<<<<')

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if args.gpus and args.gpus > 1 else None,
                         track_grad_norm=-1, max_epochs=args.epochs,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.batch_size,
                         val_check_interval=args.val_every,
                         num_sanity_val_steps=2,
                         # check_val_every_n_epoch=2,
                         limit_val_batches=args.val_percent_check,
                         limit_test_batches=args.val_percent_check,
                         logger=wandb_logger if not args.disable_checkpointing else False,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )

    if not args.test:
        trainer.fit(model)
        os.path.join(args.save_dir, args.save_prefix)
        now = datetime.now()
        save_string = now.strftime("final-model-%m/%d/%Y-%H:%M:%S")
        trainer.save_checkpoint(os.path.join(args.save_dir, args.save_prefix,save_string))
        martifact = wandb.Artifact('final_model.ckpt', type='model')
        martifact.add_file(os.path.join(args.save_dir, args.save_prefix,save_string))
        wandb_logger.experiment.log_artifact(martifact)

    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="triviaQa")
    parser = TriviaQA.add_model_specific_args(main_arg_parser, os.getcwd())
    parser = InteractiveTriviaQA.add_interactive_specific_args(parser)
    args = parser.parse_args()
    main(args)
