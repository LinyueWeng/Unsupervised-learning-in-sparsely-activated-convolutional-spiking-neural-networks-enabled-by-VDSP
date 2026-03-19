import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from pathlib import Path
from lmfit import Minimizer, Parameters, report_fit



# data_path = '..\data\ABS_03_summary_normalized.csv'

RAW_DEVICE_DATA_PATH = "data/ABS_03_summary.dat"
MODEL_CONFIGS = {
    'Ferroelectric': {
        "params":{
            'gamma': {'value': 1, 'min': 0.01, 'max': 5},
            'gamma_pn': {'value': 1.0, 'min': 0.01, 'max': 2},
            'vth': {'value': 0.35, 'min': 0.1, 'max': 1},
            'vth_pn': {'value': 1.0, 'min': 0.5, 'max': 2},
            'alpha': {'value': 1.5, 'min': 0.01, 'max': 3},
            'alpha_pn': {'value': 1.0, 'min': 0.5, 'max': 2}
        },
        'output_mapping': {
            'gamma_p': lambda p: p['gamma'],
            'gamma_d': lambda p: p['gamma'] * p['gamma_pn'],
            'theta_p': lambda p: -p['vth'],
            'theta_d': lambda p: p['vth'] * p['vth_pn'],
            'alpha_p': lambda p: p['alpha'],
            'alpha_d': lambda p: p['alpha'] * p['alpha_pn'],
        }
    },

}
# Define the model function

def memristor_Ferroelectric(w, vmem,
                   gamma=1.0,
                   gamma_pn=1.0,
                   alpha=1.0,
                   alpha_pn=1.0,
                   vth=0.1,
                   vth_pn=1
                   ):
    alpha_p = alpha
    alpha_n = alpha * alpha_pn
    vth_p = vth
    vth_n = vth * vth_pn
    gamma_p = gamma
    gamma_n = gamma * gamma_pn

    f_p = np.power(1 - w, gamma_p)
    f_n = np.power(w, gamma_n)

    cond_p = vmem < -vth_p
    cond_n = vmem > vth_n

    g_p = np.abs(np.exp(-alpha_p * (vmem + vth_p)) - 1)
    g_n = np.abs(np.exp(alpha_n * (vmem - vth_n)) - 1)

    dW = (cond_p * f_p * g_p) - (cond_n * f_n * g_n)

    W = np.clip(w + dW, 0, 1)

    dW = W - w

    return dW

class ModelCharac:
    def __init__(self, memristor_type, force_recompute=False):
        self.force_recompute = force_recompute
        data_path=self.normalize_data(RAW_DEVICE_DATA_PATH)

        self.memristor_type = memristor_type
        self.save_path = f"{data_path}/../params_{memristor_type}.csv"


        self.memristor_dictionary = {'Ferroelectric': memristor_Ferroelectric, }

        self.memristor_type = memristor_type

        data = pd.read_csv(data_path)
        data.head()
        V = data['V'].values
        dw = data['dw'].values
        w = data['w'].values

        self.Winit = w.copy()
        self.vin = V.copy()
        self.dW = dw.copy()

    def __call__(self):
        if os.path.exists(self.save_path) and not self.force_recompute:
            print(f">>> Found existing parameters for {self.memristor_type}, loading...")
            df = pd.read_csv(self.save_path)
            print(df)
            return df.to_dict(orient='records')[0]

        model_func = self.memristor_dictionary.get(self.memristor_type)
        config = MODEL_CONFIGS.get(self.memristor_type)
        params_comfig=config.get('params')
        mapping_config = config.get('output_mapping')
        if model_func and config:

            params = Parameters()
            for p_name, p_setting in params_comfig.items():
                params.add(p_name, **p_setting)

            model = Model(model_func, independent_vars=['vmem', 'w'])

            fit = model.fit(self.dW, params, vmem=self.vin, w=self.Winit)

            final_params = {name: p.value for name, p in fit.params.items()}

            if mapping_config:
                final_params = {key: func(final_params) for key, func in mapping_config.items()}

            parameter_df = pd.DataFrame([final_params])
            parameter_df.to_csv(self.save_path, index=False)
            print(f">>> Parameters saved as: {self.save_path}")
            print(fit.fit_report())
            print(final_params)
            return final_params
        else:
            print("Model not found or configuration not provided. Probably the model does not need to be characterized.")
            return None


    def normalize_data(self,file_path):
        # file_path = "..\data\ABS_03_summary.csv"

        save_name = Path(file_path).with_name(Path(file_path).stem + "_normalized.csv")
        if os.path.exists(save_name) and not self.force_recompute:
            return save_name

        suffix = Path(file_path).suffix.lower()

        if suffix == '.csv':
            data = pd.read_csv(file_path)
        elif suffix == '.dat':
            data = pd.read_csv(file_path, sep=',')

        else:
            raise ValueError("Unsupported file format. Please provide a CSV or DAT file.")


        V = data['pulseAmplitude']
        dR = data['deltaRpos(measured at +80mV)']
        Rinit = data['RposInitial']
        Rfinal = Rinit + dR

        Ginitial = 1/Rinit
        Gfinal = 1/Rfinal

        Gmin = min(Ginitial.min(), Gfinal.min())
        Gmax = max(Ginitial.max(), Gfinal.max())

        winit = (Ginitial - Gmin) / (Gmax - Gmin)
        wfinal = (Gfinal - Gmin) / (Gmax - Gmin)
        dw = wfinal - winit
        w = winit

        #Save the data in a new csv file
        data_normalized = pd.DataFrame({'V': V, 'dw': dw, 'w': w})
        data_normalized.to_csv(save_name, index=False)
        print(f">>>Normalized Data saved as: {save_name}")
        return save_name

if __name__ == "__main__":
    model_charac = ModelCharac('Ferroelectric')
    model_charac()