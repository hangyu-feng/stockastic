from data_loader import DataLoader
from model import generate_model
from config import MODEL_PATH, SELECTED_COMPANIES, TRAIN_SPLIT, FITTING_PARAMS
from tensorflow.keras.models import load_model

if __name__ == "__main__":

    NEW_MODEL = True

    loader = DataLoader()


    # one-timer
    if False:
        for comp in SELECTED_COMPANIES:
            loader.save_timeseries(comp, 'daily')

    if False:
        for comp in SELECTED_COMPANIES:
            loader.save_dataset(comp)

    # running model
    if True:
        if NEW_MODEL:
            model = generate_model()
            model.save(f"{MODEL_PATH}/unfitted")
        else:
            model = load_model(f"{MODEL_PATH}/unfitted")

        # ohlcv, indicators, open_normal, open_values, y_normalizer = loader.dataset()
        # split = int(ohlcv.shape[0] * TRAIN_SPLIT)

        # x_train = ohlcv[:split]
        # y_train = open_normal[:split]
        # x_test = ohlcv[split:]
        # y_test = open_normal[split:]
        # unscaled_y_test = open_values[split:]

        # model.fit(x=x_train, y=y_train, **FITTING_PARAMS)
        # evaluation = model.evaluate(x_test, y_test)
        # print(evaluation)
