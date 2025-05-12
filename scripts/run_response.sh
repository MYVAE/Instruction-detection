python src/prepare_gradient.py --target_text "$1"
python src/prepare_hidden_state.py
python src/classification.py