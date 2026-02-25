
import torch
class Ferroelectric:
    def __init__(self,
                 weight,
                 sf_p=1.04,
                 sf_d = 1.30,
                 gamma_p = 1.62,
                 gamma_d = 1.79,
                 alpha_p = 0.67,
                 alpha_d = 0.38,
                 theta_p = -0.55,
                 theta_d = 0.47,
                 ):
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.alpha_p = alpha_p
        self.alpha_d = alpha_d
        self.theta_p = theta_p
        self.theta_d = theta_d
        self.sf_p = sf_p
        self.sf_d = sf_d
        self.weight=weight

    def __call__(self,potential,potential_threshold,cond_pot):
        v_p=(potential/potential_threshold)*self.sf_p*self.theta_p
        v_d=(potential/potential_threshold)*self.sf_d*self.theta_d
        fp,fd=self.f_v(v_p,v_d)
        gp,gd=self.g_w(v_p,v_d)
        delta_weight_ltp=fp*gp
        delta_weight_ltd=fd*gd
        return torch.where(cond_pot,delta_weight_ltp,-delta_weight_ltd)

    def f_v(self,v_p,v_d):
        ##switching rate function##
        fp=torch.where(v_p<self.theta_p,torch.exp(-self.alpha_p*(v_p-self.theta_p))-1,torch.zeros_like(v_p))
        fd=torch.where(v_d>self.theta_d,torch.exp(self.alpha_d*(v_d-self.theta_d))-1,torch.zeros_like(v_d))
        return fp,fd

    def g_w(self,v_p,v_d):
        ##window function##
        gp=torch.where(v_p<self.theta_p,torch.pow((1-self.weight),self.gamma_p),torch.zeros_like(v_p))
        gd=torch.where(v_d>self.theta_d,torch.pow(self.weight,self.gamma_d),torch.zeros_like(v_d))
        return gp,gd


