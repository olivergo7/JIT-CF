import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss


class RobertaClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)

        # 定义7个全连接层
        self.dense1 = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense4 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense5 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense6 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense7 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size, 1)

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]  # 取<s> token（相当于[CLS]）  [batch_size, hidden_size]
        y = manual_features.float()  # [batch_size, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=-1)
        x = F.relu(self.dense1(x))  # 第一层全连接并使用ReLU激活
        x = self.dropout(x)
        x = F.relu(self.dense2(x))  # 第二层全连接
        x = self.dropout(x)
        x = F.relu(self.dense3(x))  # 第三层全连接
        x = self.dropout(x)
        x = F.relu(self.dense4(x))  # 第四层全连接
        x = self.dropout(x)
        x = F.relu(self.dense5(x))  # 第五层全连接
        x = self.dropout(x)
        x = F.relu(self.dense6(x))  # 第六层全连接
        x = self.dropout(x)
        x = F.relu(self.dense7(x))  # 第七层全连接
        x = self.dropout(x)
        x = self.out_proj_new(x)  # 输出层

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

        logits = self.classifier(outputs[0], manual_features)

        prob = torch.sigmoid(logits)
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob
