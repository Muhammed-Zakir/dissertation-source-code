# Base classification model structure (used for both student and teacher)
class BaseClassifier(nn.Module):
    """
    Base Classifier model structure for Mean Teacher.

    Combines a backbone encoder and a classification head with dropout regularization.
    """
    def __init__(self, encoder, num_classes, dropout_rate=0.5):
        super().__init__()
        self.encoder = encoder

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = encoder(dummy_input)
            encode_output_features = dummy_output.view(dummy_output.size(0), -1).size(1)     
             
        # Dropout for regularization
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(encode_output_features, num_classes)
        )
        
       # Initializing the classification head weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of the classification head"""
        for m in self.classification_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classification_head(features)
        return logits