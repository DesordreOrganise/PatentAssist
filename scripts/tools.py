
from codecarbon.external.logger import logger
from codecarbon import OfflineEmissionsTracker
import os
import time
import logging

code_Green = '\033[92m'
code_blue = '\033[94m'
code_end = '\033[0m'


def measure(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.debug(f"{code_blue}Execution time of {func.__name__}: {end - start}{code_end}")
        return result, end - start
    return wrapper


# Désactiver les logs
logger.disabled = True


def codecarbone_fr(func):
    def wrapper(*args, **kwargs):
        os.makedirs("output", exist_ok=True)
        tracker = OfflineEmissionsTracker(
            project_name="BetterCall",
            country_iso_code="FRA",
            output_dir="output",
            output_file="code_carbone_benchmark.csv"
        )
        tracker.start()

        result = func(*args, **kwargs)

        tracker.stop()
        # print(f"La fonction {func.__name__} a émis :\n\t- {tracker.final_emissions_data.emissions} kgCO2e,\n\t- CPU : {tracker.final_emissions_data.cpu_energy} kWh\n\t- GPU : {tracker.final_emissions_data.gpu_energy} kWh")

        cpu_energy = tracker.final_emissions_data.cpu_energy
        gpu_energy = tracker.final_emissions_data.gpu_energy
        emissions = tracker.final_emissions_data.emissions

        return result, (emissions, cpu_energy, gpu_energy)
    return wrapper
