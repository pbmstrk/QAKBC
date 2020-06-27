import torch
import numpy as np
from transformers import BertTokenizer, BertForQuestionAnswering


class BatchReader:

    def __init__(self, reader_model, device):
        self.reader_model = reader_model
        self.config_reader()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def config_reader(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.reader_model)
        self.model = BertForQuestionAnswering.from_pretrained(self.reader_model)
        return self

    def encode_batch(self, batch):

        """
        Inputs:
            query: str
                the question given to the question answering model
            contexts: List[str]
                the contexts from which to extract the answer
        """

        query, contexts = batch
        inputs = [(query, context) for context in contexts]
        encoding = self.tokenizer.batch_encode_plus(inputs, pad_to_max_length=True,
                                    return_tensors="pt")
        return encoding

    def todevice(self, encoding):
        for key in encoding:
            encoding[key] = encoding[key].to(self.device)

    def process_batch(self, batch):

        self.encoding = self.encode_batch(batch)
        self.encoding = self.encoding.to(self.device)

        start_scores, end_scores = self.model(**self.encoding)

        return start_scores, end_scores

    def get_span(self, inds):

        token_ids = [self.encoding['input_ids'][ind[0]][ind[1]:ind[2]+1] for ind in inds]
        spans = [" ".join(self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=True)) for token_id in token_ids]
        return spans

    def predict(self, batch, topn=5):

        start_scores, end_scores = self.process_batch(batch)

        batch_size = start_scores.size(0)
        token_number = start_scores.size(1)

        start_scores = start_scores.view(-1).softmax(0).view(*start_scores.shape)
        end_scores = end_scores.view(-1).softmax(0).view(*end_scores.shape)

        scores = np.zeros((batch_size, token_number, token_number))

        for i in range(batch_size):
            # outer product of scores
            scorespair = torch.ger(start_scores[i], end_scores[i])
            # zero out negative length and over-length span scores
            scorespair.triu_().tril_(token_number-1)

            scorespair = scorespair.detach().cpu().numpy()
            scores[i, :, :] = scorespair

        inds = np.argpartition(scores, -topn, axis=None)[-topn:]
        inds = inds[np.argsort(np.take(scores, inds))][::-1]

        inds3d = zip(*np.unravel_index(inds, scores.shape))

        return (self.get_span(inds3d), np.take(scores, inds))