import torch.nn as nn

# from dpr repository
class Reader(nn.Module):

    def __init__(self, encoder, hidden_size):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # B: batch size, D: number of documents, L: length of documents
        B, D, L = input_ids.size()
        start_logits, end_logits, relevance_logits = self._forward(input_ids.view(B * D, L),
                                                                   attention_mask.view(B * D, L))

        return start_logits.view(B, D, L), end_logits.view(B, D, L), relevance_logits.view(B, D)

    def _forward(self, input_ids, attention_mask):

        sequence_output, _pooled_output, _hidden_states = self.encoder(input_ids, None, attention_mask, output_hidden_states=True)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return start_logits, end_logits, rank_logits