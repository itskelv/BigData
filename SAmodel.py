import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class SAmodel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # 1. Base model
        self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        # 2. Language mapping
        self.language_map = {'de': 0, 'en': 1, 'es': 2, 'fr': 3, 'ja': 4, 'ch': 5}
        self.num_languages = len(self.language_map)

        # 3. Language embedding
        self.language_embeddings = nn.Embedding(
            num_embeddings=self.num_languages,
            embedding_dim=params["language_embed_dim"]
        )

        # 4. Classifier
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        input_dim = 768 + self.params["language_embed_dim"]
        hidden_dim = self.params["classifier_hidden_size"]

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.params["dropout_rate"]),
            nn.Linear(hidden_dim, self.params["num_labels"])
        )

    def forward(self, input_ids, attention_mask, labels=None, language_ids=None):

        # Convert language strings → ids (if needed)
        if isinstance(language_ids[0], str):
            language_ids = torch.tensor(
                [self.language_map[lang] for lang in language_ids],
                device=input_ids.device
            )

        # 1. XLM-R encoder
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        cls = hidden[:, 0, :]   # [B, 768]

        # 2. Language embeddings
        lang_emb = self.language_embeddings(language_ids)  # [B, dim]

        # 3. Concatenate
        features = torch.cat([cls, lang_emb], dim=-1)

        # 4. Classifier
        logits = self.classifier(features)

        # 5. Loss
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return logits
