import pandas as pd

def preprocess_text(text):
    """Clean text by stripping whitespace."""
    return text.strip()

def prepare_data(file_path,output_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    for col in ['0','1','2']:
        df[col] = df[col].apply(preprocess_text)


    prepared_data = pd.DataFrame({
        'input_text': df['0'] + " " + df['1'],
        'response_text': df['2']
    })

    prepared_data.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

if __name__ == "__main__":
    prepare_data('casual_data_windows.csv', 'prepared_data.csv')


