### TODO:HIGH: Bad translations Filter (cannot find entities?)

from functools import partial
import os
from pathlib import Path
import pandas as pd
from .preprocess import create_nyt_train_test, TYPED_ENTITY_MARKER_FUNC, filter_nyt, filter_nyt_train_test, create_nyt_train_test_tokens_format

ENG_FOR_TR = 'eng_for_tr'

downloads_folder = Path.home() / "Downloads"
def filter_nyt_eng_for_translation(df_train,df_test):
    df_train = filter_nyt(df_train, balance=True,subsample_n = 0)
    df_test = filter_nyt(df_test, balance=False,subsample_n = 0)
    return df_train,df_test

def export_nyt_no_markers(args, output_folder = downloads_folder):
    """
    Parameters:
        args - see preprocess.py::create_nyt_tokens_format remark for sample 
        output_folder - a Path (not str)

    Returns
    -------
    df with: sents - original text (no markers), em1Text, em1Type, em2Text,em2Type
    """
    args.out_entities = True
    # Because Translation is not accurate for ~40% of entity names --> we do not want to filter before translation
    args.filter_nyt = 0
    df_train, df_test = create_nyt_train_test(args)
    # We still want only 3 classes (but all samples from those classes)    
    df_train, df_test = filter_nyt_eng_for_translation(df_train,df_test)
    
    # Write .xlsx --> ready to upload to Google Translate | Documents --> copy tranlated to Datasets\New York Times Relation Extraction\translated
    pth_train = Path(args.train_data)
    train_eng_xlsx_path = pth_train.parent / ENG_FOR_TR / (pth_train.stem + '.xlsx')
    df_train.to_excel(train_eng_xlsx_path)
    
    pth_test = Path(args.test_data)
    test_eng_xlsx_path = pth_test.parent / ENG_FOR_TR / (pth_test.stem + '.xlsx')    
    df_test.to_excel(test_eng_xlsx_path)
    # Now run create_translated_train_test 
    
def process_tr_df(df_translated_xlsx_path, df_eng):
    """        
    
    train_translated_xlsx_path = downloads_folder / 'nyt_train_tr.xlsx'
    """
    df = pd.read_excel(df_translated_xlsx_path,index_col=0)
    # Restore index, columnn names, relations labels from df_eng    
    if df.shape != df_eng.shape: 
        raise Exception("translated df (df) must be the same columns.count and row count as origin df_eng")
    df.columns = df_eng.columns
    df['relations'] = df_eng.relations.to_numpy()    
    df['em1Type'] = df_eng.em1Type.to_numpy()
    df['em2Type'] = df_eng.em2Type.to_numpy()
    df['em1Text'] = df.em1Text.str.strip()
    df['em2Text'] = df.em2Text.str.strip()
    # Extract markers 
    repl_markers = partial(TYPED_ENTITY_MARKER_FUNC, trans_unicode=False)
    df['sents'] = df.apply(repl_markers, axis=1)
    df = df[~df.sents.isna()]
    df = df.drop_duplicates()
    return df
    
def create_translated_train_test(args,train_translated_xlsx_path, test_translated_xlsx_path):
    """
    Parameters:
        args - see preprocess.py::create_nyt_tokens_format remark for sample 
        train_translated_xlsx_path, test_translated_xlsx_path - paths to translated df in excel format (output of Google translate)

    Usage:
    pth_train = Path(nyt_args.train_data)
    train_translated_xlsx_path = pth_train.parent / 'translated' / (pth_train.stem + '.xlsx')
    pth_test = Path(nyt_args.test_data)    
    test_translated_xlsx_path = pth_test.parent / 'translated' / (pth_test.stem + '.xlsx')    
    Returns
    -------
    df_train, df_test with translated text, but with column headings and labels in english
    """
    # Recreate eng dataset, to copy some info to translated df (see process_tr_df)
    args.out_entities = True
   
    pth_train_trans = Path(train_translated_xlsx_path)
    df_eng_train = pd.read_excel(pth_train_trans.parent.parent / ENG_FOR_TR / pth_train_trans.name,index_col=0)
    pth_test_trans = Path(test_translated_xlsx_path)
    df_eng_test = pd.read_excel(pth_test_trans.parent.parent / ENG_FOR_TR / pth_test_trans.name, index_col=0)
    
    df_train = process_tr_df(train_translated_xlsx_path,df_eng_train)
    df_test = process_tr_df(test_translated_xlsx_path, df_eng_test)
    
    args.filter_nyt = 1000 # Now filter and balance what remained after inaccurate translation dropped
    df_train, df_test = filter_nyt_train_test(args, df_train, df_test)
    df_train, df_test, rm = create_nyt_train_test_tokens_format(df_train, df_test)
    return df_train, df_test, rm
        
    
    
    
