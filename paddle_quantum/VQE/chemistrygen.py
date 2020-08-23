# Copyright (c) 2020 Paddle Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
chemistry
"""

from numpy import array, kron, trace
import openfermion
import openfermionpyscf
import scipy
import scipy.linalg

__all__ = [
    "calc_H_rho_from_qubit_operator",
    "read_calc_H",
]


def calc_H_rho_from_qubit_operator(qubit_op, n_qubits):
    """
    Generate a Hamiltonian from QubitOperator
    Returns: H, rho
    """

    # const
    beta = 1

    sigma_table = {
        'I': array([[1, 0], [0, 1]]),
        'Z': array([[1, 0], [0, -1]]),
        'X': array([[0, 1], [1, 0]]),
        'Y': array([[0, -1j], [1j, 0]])
    }

    # calc Hamiltonian
    H = 0
    for terms, h in qubit_op.terms.items():
        basis_list = ['I'] * n_qubits
        for index, action in terms:
            basis_list[index] = action

        for sigma_symbol in basis_list:
            b = sigma_table.get(sigma_symbol, None)
            if b is None:
                raise Exception('unrecognized basis')
            h = kron(h, b)

        H = H + h

    # calc rho
    rho = scipy.linalg.expm(-1 * beta *
                            H) / trace(scipy.linalg.expm(-1 * beta * H))

    return H.astype('complex64'), rho.astype(
        'complex64')  # the returned dic will have 2 ** n value


def read_calc_H(geo_fn, multiplicity=1, charge=0):
    """
    Read and calc the H and rho
    return: H,rho matrix
    """

    if not isinstance(geo_fn, str):  # geo_fn = 'h2.xyz'
        raise Exception('filename is a string')

    geo = []

    with open(geo_fn) as f:
        f.readline()
        f.readline()

        for line in f:
            species, x, y, z = line.split()
            geo.append([species, (float(x), float(y), float(z))])

    # meanfield data
    mol = openfermion.hamiltonians.MolecularData(geo, 'sto-3g', multiplicity,
                                                 charge)
    openfermionpyscf.run_pyscf(mol)

    
    terms_molecular_hamiltonian = mol.get_molecular_hamiltonian(
    )
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(
        terms_molecular_hamiltonian)
    qubit_op = openfermion.transforms.jordan_wigner(
        fermionic_hamiltonian)


    # calc H, rho
    H, rho = calc_H_rho_from_qubit_operator(qubit_op, mol.n_qubits)
    return H, rho, mol.n_qubits


def main():
    """
    The main function
    Returns:
    """

    filename = 'h2.xyz'
    H, rho, N = read_calc_H(geo_fn=filename)
    print('H', H)
    print("--------------------------  ")
    print('rho', rho)


if __name__ == '__main__':
    main()
