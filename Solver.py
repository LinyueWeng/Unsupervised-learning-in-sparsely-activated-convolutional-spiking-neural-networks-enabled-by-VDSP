import torch
import numpy as np
from tqdm import tqdm

class CoDesignSolver:
    def __init__(self, timesteps=15):
        self.T = timesteps
        self.stats = None

    def extract_statistics(self, model, dataloader, device, num_batches=500):
        """
        Sampling Function：
        find P_spike, P_LTD, P_LTP and V_drift_norm of every timesteps
        """
        print("\n>>> [Co-design Solver]")

        all_pixels = []
        spike_counts_per_t = np.zeros(self.T)

        batch_cnt = 0
        with torch.no_grad():
            for img, _ in tqdm(dataloader, desc="Extracting statistics..."):
                if batch_cnt >= num_batches:
                    break
                img = img.to(device)

                pixels = img.flatten().cpu().numpy()
                pixels = np.clip(pixels, 0.0, 1.0)
                all_pixels.append(pixels)

                _, spike = model.forward(img, is_training=False)

                if spike.dim() == 5:
                    spikes_t = spike.sum(dim=(1,2,3,4)).cpu().numpy()
                else:
                    spikes_t = spike.sum(dim=(1,2,3)).cpu().numpy()

                spike_counts_per_t += spikes_t
                batch_cnt += 1

        total_spikes = np.sum(spike_counts_per_t) + 1e-9
        P_spike = spike_counts_per_t / total_spikes

        X = np.concatenate(all_pixels)
        t_pre = np.floor((1.0 - X) * self.T)

        P_LTD = np.zeros(self.T)
        P_LTP = np.zeros(self.T)
        V_drift_norm = np.zeros(self.T)

        for t_idx in range(self.T):
            t = t_idx + 1

            mask_ltd = t_pre >= t

            p_ltd = np.mean(mask_ltd)
            p_ltp = 1.0 - p_ltd

            P_LTD[t_idx] = p_ltd
            P_LTP[t_idx] = p_ltp

            if p_ltd > 0:
                v_pot_norm = t / (t_pre[mask_ltd] + 1.0)
                V_drift_norm[t_idx] = np.mean(v_pot_norm)
            else:
                V_drift_norm[t_idx] = 0.0

        self.stats = {
            'P_spike': P_spike,
            'P_LTD': P_LTD,
            'P_LTP': P_LTP,
            'V_drift_norm': V_drift_norm
        }

        print(">>> [Statistic feature estimation completed].")
        return self.stats

    def solve_parameter(self, target_beta=1.05, v_ref=None, sfp=None, sfd=None,
                        alpha_p=0.67, theta_p=-0.55, alpha_d=0.38, theta_d=0.47,
                        gamma_p=1.62, gamma_d=1.79, w_mean=0.25):

        if self.stats is None:
            raise ValueError("Please call extract_statistics() to extract Statistic features first!")

        params = {'v_ref': v_ref, 'sfp': sfp, 'sfd': sfd}
        missing_params = [k for k, v in params.items() if v is None]

        target_param = missing_params[0]
        print(f"\n>>> [Co-design Solver] target beta = {target_beta}")
        print(f">>> stable network weights mean: w_mean = {w_mean}")
        print(f">>> Computing the closed loop solution: {target_param.upper()} ...")

        P_s = self.stats['P_spike']
        P_d = self.stats['P_LTD']
        P_p = self.stats['P_LTP']
        V_d = self.stats['V_drift_norm']

        w_mean = w_mean * 0.99

        gp_mean = (1.0 - w_mean) ** gamma_p
        gd_mean = w_mean ** gamma_d

        def balance_error(x):
            v = v_ref if target_param != 'v_ref' else x[0]
            sp = sfp if target_param != 'sfp' else x[0]
            sd = sfd if target_param != 'sfd' else x[0]

            total_ltp_expectation = 0.0
            total_ltd_expectation = 0.0

            for t_idx in range(self.T):
                if P_s[t_idx] == 0: continue

                if sp > 1.0:
                    fp = np.exp(alpha_p * abs(theta_p) * (sp - 1.0)) - 1.0
                else:
                    fp = 0.0
                total_ltp_expectation += P_s[t_idx] * P_p[t_idx] * fp * gp_mean

                v_inner = (v - V_d[t_idx]) * sd - 1.0
                if v_inner > 0:
                    fd = np.exp(alpha_d * theta_d * v_inner) - 1.0
                else:
                    fd = 0.0
                total_ltd_expectation += P_s[t_idx] * P_d[t_idx] * fd * gd_mean

            if total_ltp_expectation <= 0: return 1e6
            if total_ltd_expectation <= 0: return -target_beta

            current_beta = total_ltd_expectation / total_ltp_expectation
            return current_beta - target_beta

        from scipy.optimize import least_squares
        bounds_dict = {'v_ref': (0.5, 3.0), 'sfp': (1.001, 8.0), 'sfd': (1.001, 8.0)}

        def objective(x):
            return [balance_error(x)]

        result = least_squares(
            objective,
            x0=[1.1],
            bounds=([bounds_dict[target_param][0]], [bounds_dict[target_param][1]  ]),
            ftol=1e-8, xtol=1e-8
        )

        if result.success and abs(result.fun[0]) < 1e-3:
            optimal_value = result.x[0]
            print(f">>> [Solution] Optimal Solution is: {target_param.upper()} = {optimal_value:.4f}")
            return optimal_value
        else:
            print(f">>> [Warning] Solver failed to converge, best approximation is: {result.x[0]:.4f}")
            return result.x[0]