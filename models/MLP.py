import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decoder
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            raise NotImplementedError
        if self.task_name == "imputation":
            raise NotImplementedError
        if self.task_name == "anomaly_detection":
            raise NotImplementedError
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection1 = nn.Linear(configs.enc_in*configs.seq_len, 256)
            self.projection2 = nn.Linear(256, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        output = x_enc.reshape(x_enc.size(0), -1)  # (batch_size, timestamps*enc_in)
        output = self.projection1(output)  # (batch_size, num_classes)
        output = self.act(output)
        output = self.dropout(output)
        output = self.projection2(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            raise NotImplementedError
        if self.task_name == "imputation":
            raise NotImplementedError
        if self.task_name == "anomaly_detection":
            raise NotImplementedError
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
