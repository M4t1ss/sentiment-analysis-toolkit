import config
import transformers
import torch.nn as nn


class BERTBaseCased(nn.Module):
    def __init__(self, DROPOUT):
        super(BERTBaseCased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, return_dict=False)

        self.bert_drop = nn.Dropout(DROPOUT)

        self.out = nn.Linear(768, 3)

        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

    def extract_features(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        return bo