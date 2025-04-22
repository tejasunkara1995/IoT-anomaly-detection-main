import argparse
import os
import torch
import torch.nn as nn
import random
import numpy as np


class LightweightLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, width_multiplier=1.0):
        super(LightweightLSTM, self).__init__()
        adjusted_hidden_size = int(hidden_size * width_multiplier)
        self.lstm = nn.LSTM(input_size, adjusted_hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(adjusted_hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out


if __name__ == '__main__':
    # For reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Arguments
    parser = argparse.ArgumentParser(description='Quantize and export model for Raspberry Pi.')
    parser.add_argument('--without_width_multiplier', action='store_true', help='Use full-size model (no multiplier)')
    parser.add_argument('--cores', type=int, default=1, help='Limit CPU threads (simulate or constrain Pi usage)')
    args = parser.parse_args()

    # Limit CPU threads for Raspberry Pi (resource control)
    torch.set_num_threads(args.cores)

    # Model settings (must match training)
    input_size = 13  # number of features
    hidden_size = 512
    output_size = 1
    num_layers = 2
    width_multiplier = 1.0 if args.without_width_multiplier else 0.5

    # Load the trained model
    model = LightweightLSTM(input_size, hidden_size, output_size, num_layers, width_multiplier)
    model_path = os.path.join("save_model", 
        'model_lstm_2023-11-22_22-07-05_without_width_multiplier.pt' 
        if args.without_width_multiplier else 
        'model_lstm_2023-11-23_11-01-34.pt'
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Apply quantization
    print("Applying quantization...")
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )

    # Save the quantized model
    export_name = "quantized_model_lstm_without_multiplier.pt" if args.without_width_multiplier else "quantized_model_lstm.pt"
    export_path = os.path.join("save_model", export_name)
    torch.save(quantized_model.state_dict(), export_path)
    print(f"✅ Quantized model saved to: {export_path}")