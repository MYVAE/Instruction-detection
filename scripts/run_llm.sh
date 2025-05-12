python src/prepare_gradient.py --model_name "$1"
python src/prepare_hidden_state.py --model_name "$1"
python src/classification.py