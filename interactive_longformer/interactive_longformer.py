import sys
import os
from collections import defaultdict
import argparse
import json
import string
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, AutoModel, AutoConfig, AutoModelWithLMHead
from scripts.triviaqa_utils import evaluation_utils

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from longformer.longformer import Longformer
from longformer.sliding_chunks import pad_to_window_size

sys.path.append(".")
from scripts.triviaqa import TriviaQADataset, TriviaQA

class InteractiveQA(TriviaQA):

    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions, answer_token_ids):
        if 'longformer' in self.args.model_path:
            question_end_index = self._get_question_end_index(input_ids)
            # Each batch is one document, and each row of the batch is a chunck of the document.
            # Make sure all rows have the same question length.
            assert (question_end_index[0].float() == question_end_index.float().mean()).item()

            # local attention everywhere
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            # global attention for the question tokens
            attention_mask[:, :question_end_index.item()] = 2

            # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
            input_ids, attention_mask = pad_to_window_size(
                input_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)

            sequence_output = self.model(
                    input_ids,
                    attention_mask=attention_mask)[0]

            # The pretrained TriviaQA model wasn't trained with padding, so remove padding tokens
            # before computing loss and decoding.
            padding_len = input_ids[0].eq(self.tokenizer.pad_token_id).sum()
            if padding_len > 0:
                sequence_output = sequence_output[:, :-padding_len]
        elif self.args.model_path in ['bart.large', 'bart.base']:
            sequence_output = self.model.extract_features(input_ids)
        else:
            if self.args.seq2seq:
                decoder_input_ids = answer_token_ids[:, 0, :-1].clone()
                decoder_input_ids[decoder_input_ids == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
                decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
                labels = answer_token_ids[:, 0, 1:].contiguous()
                labels[answer_token_ids[:, 0, 1:] == self.tokenizer.pad_token_id] = -100
                outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels)
                loss = outputs[0]
                logit_scores = outputs[1].softmax(dim=2)[:, :, 0].sum(dim=1)
                return [loss, logit_scores]
            else:
                sequence_output = self.model(input_ids, attention_mask=attention_mask)[0]

        logits = self.qa_outputs(sequence_output)
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

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = TriviaQA(args)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        # save_last=True,
        mode='min',
        period=-1,
        prefix=''
    )

    print(args)
    train_set_size = 110648  # hardcode dataset size. Needed to compute number of steps for the lr scheduler
    args.steps = args.epochs * train_set_size / (args.batch_size * max(args.gpus, 1))
    print(f'>>>>>>> #steps: {args.steps}, #epochs: {args.epochs}, batch_size: {args.batch_size * args.gpus} <<<<<<<')

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if args.gpus and args.gpus > 1 else None,
                         track_grad_norm=-1, max_epochs=args.epochs, early_stop_callback=None,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.batch_size,
                         val_check_interval=args.val_every,
                         num_sanity_val_steps=2,
                         # check_val_every_n_epoch=2,
                         val_percent_check=args.val_percent_check,
                         test_percent_check=args.val_percent_check,
                         logger=logger if not args.disable_checkpointing else False,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         show_progress_bar=not args.no_progress_bar,
                         use_amp=not args.fp32, amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="triviaQa")
    parser = TriviaQA.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
#    main(args)
