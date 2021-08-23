from tensorflow.keras.models import load_model

from data_loader import DataLoader
from model import generate_model
from config import MODEL_PATH, SELECTED_COMPANIES
from dataset import split, raw_to_dataset
from plot import plot


if __name__ == "__main__":

    loader = DataLoader()

    # for comp in SELECTED_COMPANIES:
    #     loader.save_timeseries(comp, 'daily')

    # for comp in SELECTED_COMPANIES:
    #     loader.save_dataset(comp)



    symbol = 'WMT'

    ds = raw_to_dataset(loader.get_raw(symbol))

    # ds = loader.read_dataset(symbol)
    train, validation, test = split(ds)

    model = generate_model()
    # model = load_model(f"{MODEL_PATH}/{symbol}")
    model.fit(
        x=train,
        validation_data=validation,
        epochs=30,
        shuffle=True,
        # batch_size=32
    )
    model.save(f"{MODEL_PATH}/{symbol}")

    evaluation = model.evaluate(test)
    print(evaluation)

    test_arr = [float(x[1]) for x in test.unbatch()]
    prediction = [x[-1][-1] for x in model.predict(test)]
    plot(test_arr, prediction)
