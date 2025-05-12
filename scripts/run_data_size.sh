python src/prepare_gradient.py --train_data_size "$1"
python src/prepare_hidden_state.py --train_data_size "$1"
python src/classification.py