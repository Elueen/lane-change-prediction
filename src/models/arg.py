import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(description="LANE FORECASTING")
    parser.add_argument("--model_name", type=str, default="LPLSTM", help="RNN, LSTM, GRU, LPRNN, LPLSTM, LPGRU")
    parser.add_argument("--file_path", type=str, default="outcome/files/CCG_features100.pth", help="Path to file")
    parser.add_argument("--mfi", type=str, default="resampling", help="methods for imbalanced, resampling or weight")
    parser.add_argument("--num_epoch", type=int, default=500, help="number of epoch of training")
    parser.add_argument("--min_epochs", type=int, default=100, help="minimum number of epoch of training")
    parser.add_argument("--early_stop", type=int, default=20, help="early stop step of epoch of training")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--hidden_size", type=int, default=64, help="model hidden size")
    parser.add_argument("--num_layers", type=int, default=1, help="RNN layers")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict


if __name__ == "__main__":
    command_line_args = parse_arguments()
    print("Arguments as dictionary:", command_line_args)
