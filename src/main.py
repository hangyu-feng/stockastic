from tensorflow.keras.models import load_model

from data_loader import DataLoader
from model import generate_model
from config import MODEL_PATH
from dataset import Dataset
from plot import plot


if __name__ == "__main__":

    loader = DataLoader()
    selected_companies = ['AMD', 'IBM', 'MSFT', 'PCG', 'WMT']

    # for comp in selected_companies:
    #     loader.save_timeseries(comp, 'daily')

    # for comp in selected_companies:
    #     loader.save_dataset(comp)

    symbol = 'WMT'

    raw = loader.get_raw(symbol)
    ds = Dataset(raw)

    print()

    # model = generate_model()
    # # model = load_model(f"{MODEL_PATH}/{symbol}")
    # model.fit(
    #     x=ds.train,
    #     validation_data=ds.validation,
    #     epochs=30,
    #     shuffle=True,
    #     # batch_size=32
    # )
    # model.save(f"{MODEL_PATH}/{symbol}")

    # evaluation = model.evaluate(ds.test)
    # print(evaluation)

    # test_arr = [float(x[1]) for x in ds.test.unbatch()]
    # prediction = [x[-1][-1] for x in model.predict(ds.test)]
    # plot(test_arr, prediction)
