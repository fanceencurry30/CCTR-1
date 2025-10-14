import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=100, num_heads=8, hidden_dim=768, dropout=0.1, num_encoder_layers=6, num_decoder_layers=6, temperature=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embed_dim = 512
        self.temperature = temperature

        # Truncation value prediction module
        self.truncation_predictor = nn.Sequential(
            nn.Linear(3, 64),  # Input is [std, entropy, top1_prob]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )
        # Initialize bias to ensure truncation value is initially close to 0
        self.truncation_predictor[-2].bias.data.fill_(-5.0)  # Negative bias makes sigmoid output close to 0
        self.truncation_scale = 0.01  # Scale to [0, 0.01]

        # Sequence embedding
        self.seq_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(dropout)
        )

        # Three independent Encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.query_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.key_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.value_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.bi_cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.embed_dim)

        # GRU
        self.gru = nn.GRU(
            self.embed_dim,
            self.embed_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Decoder module
        decoder_layer = nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True
            ),
            'conv1d': nn.Conv1d(self.embed_dim, self.embed_dim * 2, kernel_size=3, padding=1),
            'conv_proj': nn.Conv1d(self.embed_dim * 2, self.embed_dim, kernel_size=1),
            'conv_norm': nn.LayerNorm(self.embed_dim),
            'glu': nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.GLU(),
            ),
            'ffn': nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.Dropout(dropout)
            ),
            'norm1': nn.LayerNorm(self.embed_dim),
            'norm2': nn.LayerNorm(self.embed_dim),
            'norm3': nn.LayerNorm(self.embed_dim),
            'norm4': nn.LayerNorm(self.embed_dim)
        })
        self.decoder = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, feature_dim)
        )

    def forward(self, ocr_probs, lm_probs):
        # Input shape: [batch, 1, 100]
        batch_size = ocr_probs.size(0)

        # Calculate truncation value
        std = torch.std(ocr_probs, dim=-1, keepdim=True)  # [batch, 1, 1]
        entropy = -torch.sum(ocr_probs * torch.log(ocr_probs + 1e-10), dim=-1, keepdim=True)  # [batch, 1, 1]
        top1_prob = ocr_probs.max(dim=-1, keepdim=True)[0]  # [batch, 1, 1]
        stats = torch.cat([std, entropy, top1_prob], dim=-1)  # [batch, 1, 3]
        truncation = self.truncation_predictor(stats) * self.truncation_scale  # [batch, 1, 1], range [0, 0.01]

        # Apply truncation and re-softmax
        ocr_probs_truncated = ocr_probs - truncation
        ocr_probs_truncated = F.softmax(ocr_probs_truncated / self.temperature, dim=-1)  # [batch, 1, 100]

        # Reshape input to sequence
        ocr_seq = ocr_probs_truncated.view(batch_size, self.feature_dim, 1)  # [batch, 100, 1]
        lm_seq = lm_probs.view(batch_size, self.feature_dim, 1)  # [batch, 100, 1]

        # Sequence embedding
        ocr_seq = self.seq_embed(ocr_seq)  # [batch, 100, embed_dim]
        lm_seq = self.seq_embed(lm_seq)    # [batch, 100, embed_dim]

        # Encoder
        query_feat = self.query_encoder(lm_seq)    # [batch, 100, embed_dim]
        key_feat = self.key_encoder(ocr_seq)      # [batch, 100, embed_dim]
        value_feat = self.value_encoder(ocr_seq)  # [batch, 100, embed_dim]

        # Cross attention
        attn_output1, _ = self.cross_attention(query_feat, key_feat, value_feat)
        attn_output2, _ = self.bi_cross_attention(key_feat, query_feat, value_feat)
        attn_output = self.cross_norm((attn_output1 + attn_output2) / 2)

        # GRU
        gru_output, _ = self.gru(attn_output)
        gru_output = gru_output[:, :, :self.embed_dim] + gru_output[:, :, self.embed_dim:]

        # Decoder
        fusion = gru_output
        for layer in self.decoder:
            self_attn_out, _ = layer['self_attn'](fusion, fusion, fusion)
            fusion = layer['norm1'](fusion + self_attn_out)
            conv_out = layer['conv1d'](fusion.transpose(1, 2))
            conv_out = F.gelu(conv_out)
            conv_out = layer['conv_proj'](conv_out).transpose(1, 2)
            conv_out = layer['conv_norm'](conv_out)
            fusion = layer['norm2'](fusion + conv_out)
            glu_out = layer['glu'](fusion)
            fusion = layer['norm3'](fusion + glu_out)
            ffn_out = layer['ffn'](fusion)
            fusion = layer['norm4'](fusion + ffn_out)

        # Output layer
        fusion = fusion.mean(dim=1)  # [batch, embed_dim]
        logits = self.output(fusion)  # [batch, 100]
        fusion_probs = F.softmax(logits / self.temperature, dim=-1)  # [batch, 100]

        return fusion_probs

class ResidualPredictorOCRLM(nn.Module):
    def __init__(self, feature_dim=100, dropout=0.1, temperature=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = 128
        self.temperature = temperature

        # Truncation value prediction module
        self.truncation_predictor = nn.Sequential(
            nn.Linear(3, 64),  # Input is [std, entropy, top1_prob]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.truncation_predictor[-2].bias.data.fill_(-5.0)  # Initialize bias to make truncation value close to 0
        self.truncation_scale = 0.01

        # Input projection
        self.input_proj = nn.Linear(2 * feature_dim, self.embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Normalization
        self.norm = nn.LayerNorm(self.embed_dim)

        # MLP output: binary classification logits
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, ocr_probs, lm_probs):
        # Calculate truncation value
        std = torch.std(ocr_probs, dim=-1, keepdim=True)  # [batch, 1, 1]
        entropy = -torch.sum(ocr_probs * torch.log(ocr_probs + 1e-10), dim=-1, keepdim=True)  # [batch, 1, 1]
        top1_prob = ocr_probs.max(dim=-1, keepdim=True)[0]  # [batch, 1, 1]
        stats = torch.cat([std, entropy, top1_prob], dim=-1)  # [batch, 1, 3]
        truncation = self.truncation_predictor(stats) * self.truncation_scale  # [batch, 1, 1]

        # Apply truncation and re-softmax
        ocr_probs_truncated = ocr_probs - truncation
        ocr_probs_truncated = F.softmax(ocr_probs_truncated / self.temperature, dim=-1)  # [batch, 1, 100]

        # Input shape: [batch, 1, 100]
        input_for_coeff = torch.cat([ocr_probs_truncated.squeeze(1), lm_probs.squeeze(1)], dim=-1)  # [batch, 200]
        input_for_coeff = input_for_coeff.unsqueeze(1)  # [batch, 1, 200]

        coeff = self.input_proj(input_for_coeff)  # [batch, 1, 128]
        coeff = self.encoder(coeff)  # [batch, 1, 128]
        coeff = self.norm(coeff)  # [batch, 1, 128]
        coeff = coeff.squeeze(1)  # [batch, 128]
        logits = self.mlp(coeff)  # [batch, 2]

        # Return positive class probability as residual coefficient
        probs = F.softmax(logits / self.temperature, dim=-1)  # [batch, 2]
        residual_coeff = probs[:, 1:2]  # [batch, 1]

        return residual_coeff, logits

class ResidualPredictorOCR(nn.Module):
    def __init__(self, feature_dim=100, dropout=0.1, temperature=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = 128
        self.temperature = temperature

        # Truncation value prediction module
        self.truncation_predictor = nn.Sequential(
            nn.Linear(3, 64),  # Input is [std, entropy, top1_prob]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.truncation_predictor[-2].bias.data.fill_(-5.0)  # Initialize bias to make truncation value close to 0
        self.truncation_scale = 0.01

        # Input projection
        self.input_proj = nn.Linear(feature_dim, self.embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Normalization
        self.norm = nn.LayerNorm(self.embed_dim)

        # MLP output: binary classification logits
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, ocr_probs):
        # Calculate truncation value
        std = torch.std(ocr_probs, dim=-1, keepdim=True)  # [batch, 1, 1]
        entropy = -torch.sum(ocr_probs * torch.log(ocr_probs + 1e-10), dim=-1, keepdim=True)  # [batch, 1, 1]
        top1_prob = ocr_probs.max(dim=-1, keepdim=True)[0]  # [batch, 1, 1]
        stats = torch.cat([std, entropy, top1_prob], dim=-1)  # [batch, 1, 3]
        truncation = self.truncation_predictor(stats) * self.truncation_scale  # [batch, 1, 1]

        # Apply truncation and re-softmax
        ocr_probs_truncated = ocr_probs - truncation
        ocr_probs_truncated = F.softmax(ocr_probs_truncated / self.temperature, dim=-1)  # [batch, 1, 100]

        # Input shape: [batch, 1, 100]
        input_for_coeff = ocr_probs_truncated.squeeze(1)  # [batch, 100]
        input_for_coeff = input_for_coeff.unsqueeze(1)  # [batch, 1, 100]

        coeff = self.input_proj(input_for_coeff)  # [batch, 1, 128]
        coeff = self.encoder(coeff)  # [batch, 1, 128]
        coeff = self.norm(coeff)  # [batch, 1, 128]
        coeff = coeff.squeeze(1)  # [batch, 128]
        logits = self.mlp(coeff)  # [batch, 2]

        # Return positive class probability as residual coefficient
        probs = F.softmax(logits / self.temperature, dim=-1)  # [batch, 2]
        residual_coeff = probs[:, 1:2]  # [batch, 1]

        return residual_coeff, logits