"""Exact lossless coupled LC modal propagation without matrix exponentials.

For strictly SPD Maxwell L,C matrices, choose C=S S.T, diagonalize
S.T L S=Q diag(lambda) Q.T, and form power-reciprocal voltage/current
bases V=S^-T Q, I=S Q. Physical wave delays per length are sqrt(lambda).
Only the diagonal sin/cos propagation phase depends on frequency.

This exact stage can be cascaded with static passive R/G sections in
Strang splitting without introducing any early path before min LC delay.
"""
import numpy as np

class LosslessModal:
    def __init__(self, inductance, capacitance):
        inductance=np.asarray(inductance,float)
        capacitance=np.asarray(capacitance,float)
        n=len(inductance)
        assert inductance.shape==(n,n)==capacitance.shape
        cholesky=np.linalg.cholesky(capacitance)
        eigval,eigvec=np.linalg.eigh(cholesky.T@inductance@cholesky)
        if np.min(eigval) <= 0:
            raise ValueError("positive-definite L and C are required for lossless modal delays")
        self.delay_per_length=np.sqrt(eigval)
        self.current=cholesky@eigvec
        self.voltage=np.linalg.solve(cholesky.T,eigvec)
        # The reciprocal bases obey BV.T BI = I, conserving modal power.
        if not np.allclose(self.voltage.T@self.current,np.eye(n),rtol=1e-11,atol=1e-11):
            raise ValueError("non-power-reciprocal coupled modal bases")

    def transfer(self,s,length,zref):
        n=len(self.delay_per_length)
        prop=s*length*self.delay_per_length
        cosine=np.cosh(prop)
        sine=np.sinh(prop)
        bv,bi=self.voltage,self.current
        upper_left=(bv*cosine[None,:])@bi.T
        upper_right=-(bv*(self.delay_per_length*sine)[None,:])@bv.T/zref
        lower_left=-(bi*(sine/self.delay_per_length)[None,:])@bi.T*zref
        lower_right=(bi*cosine[None,:])@bv.T
        return np.block([[upper_left,upper_right],[lower_left,lower_right]])

    @property
    def min_delay_per_length(self):
        return float(min(self.delay_per_length))
