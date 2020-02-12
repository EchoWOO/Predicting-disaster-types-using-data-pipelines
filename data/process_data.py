# imports
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_file, categories_file):
    """
    Inputs:
    messages_file_path: string
    categories_file_path: string

    Outputs:
    df: merged message and categories
    """

    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)

    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    """
    Inputs:
    df: cleaned dataframe

    Returns:
    df pandas_dataframe: Cleaned data to be used by ML model
    """

    # Split categories
    categories = df['categories'].str.split(";",expand = True)

    # select the first row
    row = categories.iloc[0,:].values

    # extract categories
    new_cols = [r[:-2] for r in row]

    # rename the columns
    categories.columns = new_cols

    # Convert category values
    for column in categories:

        # extract the string
        categories[column] = categories[column].str[-1]

        # convert to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original column
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the split dataframe
    df[categories.columns] = categories

    # remove duplicates
    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_file_name):
    """
    Inputs
    df: the cleaned and concatenated dataframe

    Returns:
    write table to the sqlite database
    """

    # connect to dataframe
    engine = create_engine('sqlite:///{}'.format(database_file_name))
    db_file_name = database_file_name.split("/")[-1]

    #name and write table
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:
    """
    Execute all the functions above
    """

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


# run
if __name__ == '__main__':
    main()
