from Table import Table
import pandas as pd


class Table_Pair:

    def __init__(self, T: Table, T_prime: Table):
        self.T = T
        self.T_prime = T_prime
        self.sigma_A = list()
        self.sigma_A_exact = list()
        self.LHCA = list()
        self.LHDA = list()
        self.RHCA = list()
        self.RHDA = list()
        self.sigma_r = list()
        self.sigma_r_exact = list()
        self.LHCr = list()
        self.LHDr = list()
        self.RHCr = list()
        self.RHDr = list()

    def update_attribute_match(self, sigma_A):
        self.sigma_A = sigma_A
        self.LHCA = [c[0] for c in sigma_A]
        self.LHDA = list(set(self.T.A).difference(self.LHCA))
        self.RHCA = [c[1] for c in sigma_A]
        self.RHDA = list(set(self.T_prime.A).difference(self.RHCA))
        self.T.update_projected_table(self.LHCA)
        self.T_prime.update_projected_table(self.RHCA)

    def update_exact_attribute_match(self, sigma_A_exact):
        self.sigma_A_exact = sigma_A_exact

    def update_record_match(self, sigma_r):
        self.sigma_r = sigma_r
        self.LHCr = [c[0] for c in sigma_r]
        self.LHDr = list(set(self.T.r).difference(self.LHCr))
        self.RHCr = [c[1] for c in sigma_r]
        self.RHDr = list(set(self.T_prime.r).difference(self.RHCr))

    def update_exact_record_match(self, sigma_r_exact):
        self.sigma_r_exact = sigma_r_exact