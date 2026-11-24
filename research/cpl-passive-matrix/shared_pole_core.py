"""Exploratory shared-pole passive matrix characteristic admittance approximator.

The basis contains positive-real functions s/(s+a) and a/(s+a),
a>0, multiplied by rank-one PSD real symmetric matrices. Hence every
nonnegative fitted weight yields an analytic reciprocal positive-real
matrix response on the entire right Laplace half plane, not merely
at training/check frequencies. This is NOT a complete P/CPL solver:
it approximates only the LOCAL characteristic admittance, not
inter-port delayed propagation or convolution history.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvalsh, solve_triangular, sqrtm
from scipy.optimize import nnls

@dataclass
class Material:
    r: np.ndarray
    l: np.ndarray
    g: np.ndarray
    c: np.ndarray

    def validate(self):
        n = self.r.shape[0]
        for mat in [self.r,self.l,self.g,self.c]:
            assert mat.shape == (n,n)
            assert np.allclose(mat,mat.T,atol=0.,rtol=1e-14)
            assert np.min(np.linalg.eigvalsh(mat)) >= -1e-12*np.linalg.norm(mat)
        assert n >= 1
        assert np.min(np.linalg.eigvalsh(self.l)) > 0
        assert np.min(np.linalg.eigvalsh(self.c)) > 0
        return self

    def characteristic(self, s):
        if np.isinf(s):
            series, shunt = self.l, self.c
        elif s == 0 and np.max(np.abs(self.r)) == 0 and np.max(np.abs(self.g)) == 0:
            return self.characteristic(np.inf)
        else:
            series, shunt = self.r+s*self.l, self.g+s*self.c
        if np.isreal(s):
            factor = np.linalg.cholesky(series)
            composite = factor.T@shunt@factor
            eigenvalues, eigenvectors = np.linalg.eigh(composite)
            tolerance = 1e-13 * np.max(np.abs(eigenvalues))
            if eigenvalues[0] < -tolerance:
                raise ValueError("nonpassive Maxwell characteristic matrix")
            square_root = (eigenvectors*np.sqrt(np.maximum(eigenvalues,0)))@eigenvectors.T
            inv = solve_triangular(factor,np.eye(len(factor)),lower=True)
            return inv.T@square_root@inv
        value = np.linalg.solve(series,sqrtm(series@shunt))
        return (value+value.T)/2

    def timescales(self):
        rates = []
        for diag, dynamic in [(self.r,self.l),(self.g,self.c)]:
            values=eigvalsh(diag,dynamic)
            rates.extend(float(v) for v in values if np.isfinite(v) and v>0)
        if not rates:
            return 1.,1.
        return min(rates), max(rates)

def unit_vectors(model, rates, extra_pairs=False):
    n=len(model.r)
    vectors=[vec for vec in np.eye(n)]
    for s in rates:
        try:
            _, frame = np.linalg.eigh(model.characteristic(s))
            vectors.extend(frame.T)
        except np.linalg.LinAlgError:
            continue
    if extra_pairs:
        for i in range(n):
            for j in range(i+1,n):
                # For 8 conductors keep adjacent and alternating neighbours
                # in the explicit atom set; spectral frames supply dense
                # directions spanning the rest.
                if n > 4 and not (j-i == 1 or j-i == n-1):
                    continue
                for sign in (-1.,1.):
                    vec=np.eye(n)[i]+sign*np.eye(n)[j]
                    vectors.append(vec/math.sqrt(2.))
    return np.array([np.outer(v,v) for v in vectors])

class PassiveFit:
    def __init__(self, poles, atoms, coefficients, normalizer):
        self.poles=np.asarray(poles)
        self.atoms=np.asarray(atoms)
        self.coefficients=np.asarray(coefficients).reshape(2*len(poles),len(atoms))
        self.normalizer=normalizer
        assert np.min(self.coefficients) >= 0.

    def response(self,s):
        poles=self.poles
        if np.isinf(s):
            factors=np.r_[np.ones(len(poles)),np.zeros(len(poles))]
        else:
            factors=np.r_[s/(s+poles),poles/(s+poles)]
        return np.einsum("j,ja,arc->rc",factors,self.coefficients,self.atoms)*self.normalizer

def fit_passive(model, pole_count=10, samples_per_axis=44, include_imaginary=True, pair_atoms=True, low_margin=1e3, high_margin=1e3):
    model.validate()
    tstart=time.perf_counter()
    low,high=model.timescales()
    lo,hi=low/low_margin,high*high_margin
    poles=np.geomspace(lo,hi,pole_count)
    basis_rates=[0., np.inf, low, math.sqrt(low*high),high]
    atoms=unit_vectors(model,basis_rates,extra_pairs=pair_atoms)
    n=len(model.r)
    tri=np.triu_indices(n)
    a_upper=atoms[:,tri[0],tri[1]]
    unit=max(np.linalg.norm(model.characteristic(0.)),np.linalg.norm(model.characteristic(np.inf)))
    real_rates=np.r_[0.,np.geomspace(lo/10,hi*10,samples_per_axis),np.inf]
    sources = list(real_rates)
    if include_imaginary:
        sources += [1j*w for w in np.geomspace(lo/10,hi*10,samples_per_axis)]
    features = np.empty((len(sources),2*pole_count),dtype=complex)
    expected = np.empty((len(sources),len(tri[0])),dtype=complex)
    for i,s in enumerate(sources):
        if np.isinf(s):
            features[i] = np.r_[np.ones(pole_count),np.zeros(pole_count)]
        else:
            features[i] = np.r_[s/(s+poles),poles/(s+poles)]
        expected[i]=model.characteristic(s)[tri]/unit
    cols=2*pole_count*len(atoms)
    rows=[]
    rhs=[]
    for i,s in enumerate(sources):
        matrix=(features[i,:,None,None]*a_upper[None,:,:]).reshape(cols,len(tri[0])).T
        weight=10. if i in (0,len(real_rates)-1) else 1.
        rows.append(matrix.real*weight)
        rhs.append(expected[i].real*weight)
        if not np.isreal(s):
            rows.append(matrix.imag*weight)
            rhs.append(expected[i].imag*weight)
    a = np.concatenate(rows)
    b = np.concatenate(rhs)
    weights, residual=nnls(a,b,maxiter=10000)
    fit=PassiveFit(poles,atoms,weights,unit)
    return fit,{"poles":len(poles),"atoms":len(atoms),"candidates":cols,
        "active":int((weights>1e-13).sum()),
        "elapsed_s":time.perf_counter()-tstart,
        "scaled_residual":float(residual),
        "band_start":float(lo),"band_end":float(hi)}

def evaluate(model, fit):
    lo,hi=model.timescales()
    rates=np.geomspace(lo/1e4,hi*1e4,105)
    errors_real=[]
    errors_complex=[]
    passivity=[]
    reference_passivity=[]
    for rate in rates:
        for arg,collection in [(rate,errors_real),(1j*rate,errors_complex)]:
            exact=model.characteristic(arg)
            approx=fit.response(arg)
            collection.append(float(np.linalg.norm(approx-exact)/np.linalg.norm(exact)))
        herm=(fit.response(1j*rate)+fit.response(1j*rate).conj().T)/2.
        passivity.append(float(np.min(np.linalg.eigvalsh(herm))))
        original=model.characteristic(1j*rate)
        ref=(original+original.conj().T)/2
        reference_passivity.append(float(np.min(np.linalg.eigvalsh(ref))))
    return {
        "max_real_error":max(errors_real),
        "max_imag_error":max(errors_complex),
        "p95_imag_error":float(np.quantile(errors_complex,.95)),
        "min_passive_eigen":min(passivity),
        "min_reference_eigen":min(reference_passivity),
        "positive_real_analytic_proof":"sum of nonnegative scalar PR responses times PSD matrices"
    }
