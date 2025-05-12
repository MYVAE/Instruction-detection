python src/prepare_gradient.py --data_type "$1" --instruction_type "$2"
python src/prepare_hidden_state.py --data_type "$1" --instruction_type "$2"
python src/classification.py