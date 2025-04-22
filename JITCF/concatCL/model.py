import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss


class RobertaClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

     def __init__(self, config, args):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()  # ReLU activation
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.args = args

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)  # Apply ReLU activation
        x = self.dense2(x)
        x = self.relu(x)  # Apply ReLU activation again
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None):
        outputs = self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None

        # 提取 [CLS] 特征用于对比损失
        cls_feature = outputs[0][:, 0, :]  # shape: [batch_size, hidden_size]

        logits = self.classifier(outputs[0], manual_features)
        prob = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCELoss()
            loss_cls = loss_fct(prob, labels.unsqueeze(1).float())
            return loss_cls, prob, last_layer_attn_weights, cls_feature
        else:
            return prob, cls_feature
