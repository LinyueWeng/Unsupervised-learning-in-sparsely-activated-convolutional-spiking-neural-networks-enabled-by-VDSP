from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


class HHNeuron:
    ## HH Model for biological neuron, modeled as a reference for other models##
    def __init__(self,
                v=np.arange(-100, 100),# range of voltage values of interest
                E_Na = 120,
                E_K = -12,
                E_L = 10.6,
                g_Na = 120,
                g_K = 36,
                g_L = 0.3,
                C = 1,
                m_exponent=3,#here you could change the exponents of gating variables to extend the standard hh model to other approximated models
                n_exponent=4,
                h_exponent=1,
                show_gating_variables=False,# modify this to plot gating variables: asymptotic values (x0) and time constants (tau)
                show_neuron_dynamics = False):# modify this to plot neuron dynamics
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.C = C
        self.show_gating_variables = show_gating_variables
        self.show_neuron_dynamics = show_neuron_dynamics
        self.m_exponent = m_exponent
        self.n_exponent = n_exponent
        self.h_exponent = h_exponent

        self.v = v

    #neuron parameters (different types of neurons have different parameters)
    def alpha_n(self,v):
        v = np.where(v == 10, 10.0 + 1e-12, v)
        return 0.01 * (10 - v) / (np.exp((10 - v) / 10) - 1)
    def beta_n(self,v):
        return 0.125 * np.exp(-v / 80)
    def alpha_m(self,v):
        v = np.where(v == 25, 25.0 + 1e-12, v)
        return 0.1 * (25 - v) / (np.exp((25 - v) / 10) - 1)
    def beta_m(self,v):
        return 4 * np.exp(-v / 18)
    def alpha_h(self,v):
        return 0.07 * np.exp(-v / 20)
    def beta_h(self,v):
        return 1 / (np.exp((30 - v) / 10) + 1)

    #gating variables
    def tau_m(self,v):
        return 1 / (self.alpha_m(v) + self.beta_m(v))
    def tau_n(self,v):
        return 1 / (self.alpha_n(v) + self.beta_n(v))
    def tau_h(self,v):
        return 1 / (self.alpha_h(v) + self.beta_h(v))
    def x0_m(self,v):
        return self.tau_m(v) * self.alpha_m(v)
    def x0_n(self,v):
        return self.tau_n(v) * self.alpha_n(v)
    def x0_h(self,v):
        return self.tau_h(v) * self.alpha_h(v)

    # derivatives of gating variables
    def f_V(self,v, m, n, h, i):
        return (i - (self.g_L * (v - self.E_L) + self.g_K * n ** self.n_exponent * (v - self.E_K) + self.g_Na * m ** self.m_exponent * h **self.h_exponent* (v - self.E_Na))) / self.C
    def f_m(self,v, m):
        return (self.x0_m(v) - m) / self.tau_m(v)
    def f_n(self,v, n):
        return (self.x0_n(v) - n) / self.tau_n(v)
    def f_h(self,v, h):
        return (self.x0_h(v) - h) / self.tau_h(v)

    def hh_model(self,y, t):
        V, m, n, h = y
        I = input_current_function(t)
        dm = self.f_m(V, m)
        dn = self.f_n(V, n)
        dh = self.f_h(V, h)
        dV = self.f_V(V, m, n, h, I)
        return [dV, dm, dn, dh]

    def __call__(self):

        V0 = 0#initial membrane potential, change this to see how the model behaves at different initial conditions
        # observed time points, if you want to change the time points, change the input current function as well#
        dt = 0.01
        t_end = 200
        t = np.linspace(0, t_end, int(t_end / dt))

        I = np.array([input_current_function(time_point) for time_point in t])
        y0 = [V0, self.x0_m(V0), self.x0_n(V0), self.x0_h(V0)]
        solution = odeint(self.hh_model, y0, t)
        V, m, n, h = solution.T
        if self.show_gating_variables:
            # Plot asymptotic values
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(self.v, self.x0_m(self.v), label='x0_m', linewidth=2)
            ax1.plot(self.v, self.x0_n(self.v), label='x0_n', linewidth=2)
            ax1.plot(self.v, self.x0_h(self.v), label='x0_h', linewidth=2)
            ax1.set_xlabel('Voltage (mV)')
            ax1.set_ylabel('Asymptotic value')
            ax1.set_title('Asymptotic values of gating variables')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot time constants
            ax2.plot(self.v, self.tau_m(self.v), label='tau_m', linewidth=2)
            ax2.plot(self.v, self.tau_n(self.v), label='tau_n', linewidth=2)
            ax2.plot(self.v, self.tau_h(self.v), label='tau_h', linewidth=2)
            ax2.set_xlabel('Voltage (mV)')
            ax2.set_ylabel('Time constant (ms)')
            ax2.set_title('Time constants of gating variables')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        if self.show_neuron_dynamics:
            fig, axes = plt.subplots(5, 1, figsize=(12, 14))
            # Plot 1: Input current
            axes[0].plot(t, I, 'k', linewidth=2)
            axes[0].set_ylabel('Current (μA/cm²)')
            axes[0].set_title('Input Current')
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Voltage
            axes[1].plot(t, V, linewidth=2)
            axes[1].set_ylabel('Voltage (mV)')
            axes[1].set_title('Membrane Potential')
            axes[1].grid(True, alpha=0.3)

            # Plot 3: Gating variables
            axes[2].plot(t, m, label='m', linewidth=2)
            axes[2].plot(t, n, label='n', linewidth=2)
            axes[2].plot(t, h, label='h', linewidth=2)
            axes[2].set_ylabel('Gating variable')
            axes[2].set_title('Gating Variables')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Plot 4: Conductance
            axes[3].plot(t, self.g_Na * m ** 3 * h, label='g_Na', linewidth=2)
            axes[3].plot(t, self.g_K * n ** 4, label='g_K', linewidth=2)
            axes[3].set_ylabel('Conductance (mS/cm²)')
            axes[3].set_title('Ion Channel Conductance')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

            # Plot 5: Currents
            axes[4].plot(t, self.g_L * (V - self.E_L) + self.g_Na * m ** 3 * h * (V - self.E_Na) + self.g_K * n ** 4 * (V - self.E_K), label='I_L',
                         linewidth=2)
            axes[4].plot(t, self.g_Na * m ** 3 * h * (V - self.E_Na), label='I_Na', linewidth=2)
            axes[4].plot(t, self.g_K * n ** 4 * (V - self.E_K), label='I_K', linewidth=2)
            axes[4].set_xlabel('Time (ms)')
            axes[4].set_ylabel('Current (μA/cm²)')
            axes[4].set_title('Ion Channel Currents')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return V

class CableModel:
    def __init__(self,
                 r_i,# intracellular resistivity in ohm*cm
                 r_m,# membrane resistivity in ohm*cm^2
                 l = 5,# length of the cable section in cm
                 dx = 0.01, # spatial step in cm
                 v0 = 1,
                 is_show=False):# Voltage the cable input

        self.r_i = r_i
        self.r_m = r_m
        self.l = l
        self.dx = dx
        self.v0 = v0
        self.num_sections = len(r_i)
        self.is_show = is_show

    def __call__(self):
        v_ends= np.zeros(len(self.r_i)+1)
        v_ends[0]=self.v0
        v=np.zeros([self.num_sections,len(np.arange(self.l/self.dx))])
        for i in range(len(self.r_i)):
            v_sections=(v_ends[i]*self.v_x(np.arange(0, self.l, self.dx), self.r_i[i], self.r_m[i]))
            v_ends[i+1]=v_sections[-1]
            v[i]=v_sections
        v=np.concatenate(v)
        if self.is_show:
            x_values = np.arange(0, self.num_sections * self.l, self.dx)
            plt.figure()
            plt.plot(x_values, v, linewidth=2)
            plt.xlabel('Distance along the cable (cm)')
            plt.ylabel('Voltage (mV)')
            plt.grid(True, alpha=0.3)
            plt.show()
        return v

    def v_x(self,x, r_i, r_m):
        lam = (r_m / r_i) ** 0.5
        return np.exp(-x / lam)


def input_current_function(t):
    ## input current function for the neuron##
    if 2 <= t <= 2.5:
        return 10.0
    elif 10 <= t <= 10.5:
        return 20.0
    elif 31.6 <= t <= 33:
        return 10.0
    elif 33 < t <= 34:
        return -40.0
    elif 34 < t <= 35:
        return 10.0
    elif 45 <= t <= 45.5:
        return 30.0
    elif 46 <= t <= 46.5:
        return 30.0
    elif 47 <= t <= 47.5:
        return 30.0
    elif 48 <= t <= 48.5:
        return 30.0
    elif 50 <= t <= 80:
        return 30.0
    elif 110 <= t <= 140:
        return 100.0
    elif 170 <= t <= 200:
        return 200.0
    else:
        return 0.0


model=HHNeuron(show_neuron_dynamics=True)
#model=CableModel([10,20,5,10],[10000,1000,1000,1000],l=10,is_show=True)
model()