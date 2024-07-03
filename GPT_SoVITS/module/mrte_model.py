# This is Multi-reference timbre encoder

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm
from module.attentions import MultiHeadAttention


class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels=256,
        n_heads=4,
    ):
        super(MRTE, self).__init__()
        self.cross_attention = MultiHeadAttention(
            content_enc_channels, content_enc_channels, n_heads
        )

    def forward(self, ssl, ssl_mask, text_feature, text_feature_mask, ge):
        attn_mask = ssl_mask.unsqueeze(-1) * text_feature_mask.unsqueeze(2)
        ssl = (
            self.cross_attention(
                ssl * ssl_mask, text_feature * text_feature_mask, attn_mask
            )
            + ssl
            + ge
        )
        return ssl


class SpeakerEncoder(torch.nn.Module):
    def __init__(
        self,
        mel_n_channels=80,
        model_num_layers=2,
        model_hidden_size=256,
        model_embedding_size=256,
    ):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            mel_n_channels, model_hidden_size, model_num_layers, batch_first=True
        )
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels.transpose(-1, -2))
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)


class MELEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        # print(x.shape,x_lengths.shape)
        x = self.pre(x)
        x = self.enc(x)
        x = self.proj(x)
        return x


class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer)
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            acts = fused_add_tanh_sigmoid_multiply(x_in, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output

    def remove_weight_norm(self):
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input, n_channels):
    n_channels_int = n_channels[0]
    t_act = torch.tanh(input[:, :n_channels_int, :])
    s_act = torch.sigmoid(input[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


if __name__ == "__main__":
    content_enc = torch.randn(3, 192, 100)
    content_mask = torch.ones(3, 1, 100)
    ref_mel = torch.randn(3, 128, 30)
    ref_mask = torch.ones(3, 1, 30)
    model = MRTE()
    out = model(content_enc, content_mask, ref_mel, ref_mask)
    print(out.shape)
