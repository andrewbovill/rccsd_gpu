import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf import cc
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf.mp import mp2
from pyscf import __config__

import cupy as cp



def _gpu_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = cp.asarray(eris.fock[:nocc, :nocc])
    eris_ovov = cp.asarray(eris.ovov)
    Fki = 2*cp.einsum('kcld,ilcd->ki', eris_ovov, t2)
    Fki -= cp.einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Fki += 2*cp.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Fki -= cp.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    Fki += foo
    return Fki


def _gpu_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = cp.asarray(eris.fock[nocc:, nocc:])
    eris_ovov = cp.asarray(eris.ovov)
    Fac = -2*cp.einsum('kcld,klad->ac', eris_ovov, t2)
    Fac += cp.einsum('kdlc,klad->ac', eris_ovov, t2)
    Fac -= 2*cp.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Fac += cp.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    Fac += fvv
    return Fac


def _gpu_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = cp.asarray(eris.fock[:nocc, nocc:])
    eris_ovov = cp.asarray(eris.ovov)
    Fkc = 2*cp.einsum('kcld,ld->kc', eris_ovov, t1)
    Fkc -= cp.einsum('kdlc,ld->kc', eris_ovov, t1)
    Fkc += fov
    return Fkc

# Eqs. (40)-(41) "lambda"


def _gpu_Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = cp.asarray(eris.fock[:nocc, nocc:])
    Lki = _gpu_Foo(t1, t2, eris) + cp.einsum('kc,ic->ki', fov, t1)
    eris_ovoo = cp.asarray(eris.ovoo)
    Lki += 2*cp.einsum('lcki,lc->ki', eris_ovoo, t1)
    Lki -= cp.einsum('kcli,lc->ki', eris_ovoo, t1)
    return Lki


def _gpu_Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = cp.asarray(eris.fock[:nocc, nocc:])
    Lac = _gpu_Fvv(t1, t2, eris) - cp.einsum('kc,ka->ac', fov, t1)
    eris_ovvv = cp.asarray(eris.get_ovvv())
    Lac += 2*cp.einsum('kdac,kd->ac', eris_ovvv, t1)
    Lac -= cp.einsum('kcad,kd->ac', eris_ovvv, t1)
    return Lac


def _gpu_Woooo(t1, t2, eris):
    eris_ovoo = cp.asarray(eris.ovoo)
    Wklij = cp.einsum('lcki,jc->klij', eris_ovoo, t1)
    Wklij += cp.einsum('kclj,ic->klij', eris_ovoo, t1)
    eris_ovov = cp.asarray(eris.ovov)
    Wklij += cp.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += cp.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    Wklij += cp.asarray(eris.oooo).transpose(0, 2, 1, 3)
    return Wklij


def _gpu_Wvvvv(t1, t2, eris):
    # Incore
    eris_ovvv = cp.asarray(eris.get_ovvv())
    Wabcd = cp.einsum('kdac,kb->abcd', eris_ovvv, -t1)
    Wabcd -= cp.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    Wabcd += cp.asarray(imd._get_vvvv(eris)).transpose(0, 2, 1, 3)
    return Wabcd


def _gpu_Wvoov(t1, t2, eris):
    eris_ovvv = cp.asarray(eris.get_ovvv())
    eris_ovoo = cp.asarray(eris.ovoo)
    Wakic = cp.einsum('kcad,id->akic', eris_ovvv, t1)
    Wakic -= cp.einsum('kcli,la->akic', eris_ovoo, t1)
    Wakic += cp.asarray(eris.ovvo).transpose(2, 0, 3, 1)
    eris_ovov = cp.asarray(eris.ovov)
    Wakic -= 0.5*cp.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*cp.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -= cp.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic += cp.einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic


def _gpu_Wvovo(t1, t2, eris):
    eris_ovvv = cp.asarray(eris.get_ovvv())
    eris_ovoo = cp.asarray(eris.ovoo)
    Wakci = cp.einsum('kdac,id->akci', eris_ovvv, t1)
    Wakci -= cp.einsum('lcki,la->akci', eris_ovoo, t1)
    Wakci += cp.asarray(eris.oovv).transpose(2, 0, 3, 1)
    eris_ovov = cp.asarray(eris.ovov)
    Wakci -= 0.5*cp.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -= cp.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci


def update_amps(cc, t1, t2, eris):
    print("JAMES' GPU CCSD")
    print("JAMES' GPU CCSD")
    print("JAMES' GPU CCSD")
    print("JAMES' GPU CCSD")
    print("JAMES' GPU CCSD")
    t1 = cp.asarray(t1)
    t2 = cp.asarray(t2)
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = cp.asarray(fock[:nocc, nocc:].copy())
    foo = cp.asarray(fock[:nocc, :nocc].copy())
    fvv = cp.asarray(fock[nocc:, nocc:].copy())

    Foo = _gpu_Foo(t1, t2, eris)
    Fvv = _gpu_Fvv(t1, t2, eris)
    Fov = _gpu_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= cp.asarray(mo_e_o)
    Fvv[np.diag_indices(nvir)] -= cp.asarray(mo_e_v)

    # T1 equation
    t1new = -2*cp.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new += cp.einsum('ac,ic->ia', Fvv, t1)
    t1new += -cp.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*cp.einsum('kc,kica->ia', Fov, t2)
    t1new += -cp.einsum('kc,ikca->ia', Fov, t2)
    t1new += cp.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*cp.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new += -cp.einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = cp.asarray(eris.get_ovvv())
    t1new += 2*cp.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new += -cp.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*cp.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new += -cp.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    eris_ovoo = cp.asarray(eris.ovoo, order='C')
    t1new += -2*cp.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new += cp.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new += -2*cp.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new += cp.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    # T2 equation
    tmp2 = cp.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += cp.asarray(eris_ovvv).conj().transpose(1, 3, 0, 2)
    tmp = cp.einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose(1, 0, 3, 2)
    tmp2 = cp.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1, 3, 0, 2).conj()
    tmp = cp.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    t2new += cp.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

    Loo = _gpu_Loo(t1, t2, eris)
    Lvv = _gpu_Lvv(t1, t2, eris)
    Loo[np.diag_indices(nocc)] -= cp.asarray(mo_e_o)
    Lvv[np.diag_indices(nvir)] -= cp.asarray(mo_e_v)

    Woooo = _gpu_Woooo(t1, t2, eris)
    Wvoov = _gpu_Wvoov(t1, t2, eris)
    Wvovo = _gpu_Wvovo(t1, t2, eris)
    Wvvvv = _gpu_Wvvvv(t1, t2, eris)

    tau = t2 + cp.einsum('ia,jb->ijab', t1, t1)
    t2new += cp.einsum('klij,klab->ijab', Woooo, tau)
    t2new += cp.einsum('abcd,ijcd->ijab', Wvvvv, tau)
    tmp = cp.einsum('ac,ijcb->ijab', Lvv, t2)
    t2new += (tmp + tmp.transpose(1, 0, 3, 2))
    tmp = cp.einsum('ki,kjab->ijab', Loo, t2)
    t2new -= (tmp + tmp.transpose(1, 0, 3, 2))
    tmp = 2*cp.einsum('akic,kjcb->ijab', Wvoov, t2)
    tmp -= cp.einsum('akci,kjcb->ijab', Wvovo, t2)
    t2new += (tmp + tmp.transpose(1, 0, 3, 2))
    tmp = cp.einsum('akic,kjbc->ijab', Wvoov, t2)
    t2new -= (tmp + tmp.transpose(1, 0, 3, 2))
    tmp = cp.einsum('bkci,kjac->ijab', Wvovo, t2)
    t2new -= (tmp + tmp.transpose(1, 0, 3, 2))

    eia = mo_e_o[:, None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= cp.asarray(eia)
    t2new /= cp.asarray(eijab)

    return t1new.get(), t2new.get()


class RCCSD_GPU(cc.rccsd.RCCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities
    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.
        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if mbpt2:
            pt = mp2.MP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = np.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        return ccsd.CCSD.ccsd(self, t1, t2, eris)

    update_amps = update_amps
