import utils

def main():
    # Load the dataset
    train_df = utils.load_dataset('train')
    # test_df = utils.load_dataset('test')


    # Preprocess the text
    # Apply the cleaning function to both training and test data
    train_df['text'] = train_df['text'].apply(utils.clean_text)
    # test_df['text'] = test_df['text'].apply(utils.clean_text)


    # Featurization
    X_train, y_train, X_test, y_test = utils.featurize_text(train_df)
    # x_test = utils.featurize_text('test', test_df)

    # Model training
    model = utils.train_model(X_train, y_train)

    # Model evaluation
    utils.model_evaluation(model,X_test,y_test)