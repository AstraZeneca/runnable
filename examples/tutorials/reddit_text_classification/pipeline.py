from examples.tutorials.reddit_text_classification.steps import (
    clean,
    extract_text,
    model_fit,
    tfidf,
    tokenize,
)
from runnable import Pipeline, Stub, pickled


def driver():
    x, labels = extract_text(
        url="https://raw.githubusercontent.com/axsauze/reddit-classification-exploration/master/data/reddit_train.csv",
        encoding="ISO-8859-1",
        features_column="BODY",
        labels_column="REMOVED",
    )

    cleaned_x = clean(x)
    tokenised_x = tokenize(cleaned_x)
    vectorised_x = tfidf(tokenised_x, max_features=1000, ngram_range=3)
    y_probabilities = model_fit(vectorised_x, labels, c_param=0.1)

    print(y_probabilities)


def runnable_pipeline():
    extract_task = Stub(
        name="extract", function=extract_text, returns=[pickled("x"), pickled("labels")]
    )
    clean_task = Stub(name="clean", function=clean, returns=[pickled("cleaned_x")])
    tokenize_task = Stub(
        name="tokenize", function=tokenize, returns=[pickled("tokenised_x")]
    )
    vectorise_task = Stub(
        name="tfidf", function=tfidf, returns=[pickled("vectorised_x")]
    )

    model_fit_task = Stub(
        name="model_fit",
        function=model_fit,
        returns=[pickled("y_probabilities"), pickled("lr_model")],
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[extract_task, clean_task, tokenize_task, vectorise_task, model_fit_task],
        add_terminal_nodes=True,
    )

    pipeline.execute(
        parameters_file="examples/tutorials/reddit_text_classification/parameters.yaml"
    )

    return pipeline


if __name__ == "__main__":
    # driver()
    runnable_pipeline()
