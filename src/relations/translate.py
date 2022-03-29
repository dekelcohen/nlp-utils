import os
from pathlib import Path
from .preprocess import create_nyt_train_test
def export_nyt_no_markers(args, output_folder = str(Path.home() / "Downloads")):
    """
    Parameters:
        args - see preprocess.py::create_nyt_tokens_format remark for sample 

    Returns
    -------
    df with: sents - original text (no markers), em1Text, em1Type, em2Text,em2Type
    """
    args.out_entities = True
    df_train, df_test = create_nyt_train_test(args)
    df_train.to_excel(Path(output_folder) / 'nyt_train.xlsx')
    df_test.to_excel(Path(output_folder) / 'nyt_test.xlsx')
    
    
def create_translated_train_test(train_translated_xlsx_path, test_translated_xlsx_path):
    pass
    