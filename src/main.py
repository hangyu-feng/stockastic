from data_loader import DataLoader
from model import model
from config import SELECTED_COMPANIES

if __name__ == "__main__":

    loader = DataLoader()

    # one-timer
    if False:
        for comp in SELECTED_COMPANIES:
            loader.save_timeseries_daily(comp)
