import copy
import numpy as np
import sympy as sp

def deformation_gradient(shear, gamma):
    """
        Calculates the deformation gradient F associated with a given type of shear for symbolic F.
        :param shear: type of shear ('fs', 'sf', 'sn', 'ns', 'fn' or 'nf')
        :param gamma: amount of shear
        :return: deformation gradient F
    """
    # translate axes f, s and n to indices
    axes_to_indices = {"f":0, "s":1, "n":2}
    ind1 = axes_to_indices[shear[0]]
    ind2 = axes_to_indices[shear[1]]

    # calculate deformation gradient
    F = sp.eye(3)
    F[ind2, ind1] = gamma
    return F

def deformation_gradient_np(shear, gamma):
    """
        Calculates the deformation gradient F associated with a given type of shear using numpy modules.
        :param shear: type of shear ('fs', 'sf', 'sn', 'ns', 'fn' or 'nf')
        :param gamma: amount of shear
        :return: deformation gradient F
    """
    # translate axes f, s and n to indices
    axes_to_indices = {"f":0, "s":1, "n":2}
    ind1 = axes_to_indices[shear[0]]
    ind2 = axes_to_indices[shear[1]]

    # calculate deformation gradient
    F = np.identity(3)
    F[ind2, ind1] = gamma
    return F


def invariants(C):
    """
    Calculates the invariants corresponding to a Symbolic Cauchy-Green tensor C.
    :param C: right Cauchy-Green tensor
    :return: invariants I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns
    """

    # calculate basis vectors
    e_f0 = sp.Matrix([[1], [0], [0]])
    e_s0 = sp.Matrix([[0], [1], [0]])
    e_n0 = sp.Matrix([[0], [0], [1]])

    # calculate invariants I1, I2, I3
    I1 = sp.trace(C)
    I2 = (I1**2 - sp.trace(C**2)) / 2
    I3 = sp.det(C)

    # calculate invariants I4 and I5 for the different axes f, s and n
    I4f = (e_f0.T * (C*e_f0))[0, 0]
    I4s = (e_s0.T * (C*e_s0))[0, 0]
    I4n = (e_n0.T * (C*e_n0))[0, 0]
    I5f = (e_f0.T * (C**2*e_f0))[0, 0]
    I5s = (e_s0.T * (C**2*e_s0))[0, 0]
    I5n = (e_n0.T * (C**2*e_n0))[0, 0]

    # calculate the coupling invariants I8 (note: I8ij = I8ji)
    I8fs = (e_f0.T * (C*e_s0))[0, 0]
    I8fn = (e_f0.T * (C*e_n0))[0, 0]
    I8ns = (e_n0.T * (C*e_s0))[0, 0]

    return I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns



def invariants_np(F):
    """
    Calculates the invariants corresponding to the deformation gradient F using numpy modules.
    :param F: deformation gradient
    :return: invariants I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns
    """

    # calculate right Cauchy-Green Tensor C
    C = np.dot(F.T, F)

    # calculate basis vectors
    e_f0 = np.array([[1], [0], [0]])
    e_s0 = np.array([[0], [1], [0]])
    e_n0 = np.array([[0], [0], [1]])

    # calculate invariants I1, I2, I3
    I1 = np.trace(C)
    I2 = (I1**2 - np.trace(np.linalg.matrix_power(C, 2))) / 2
    I3 = np.linalg.det(C)

    # calculate invariants I4 and I5 for the different axes f, s and n
    I4f = np.dot(e_f0.T, np.dot(C, e_f0))
    I4s = np.dot(e_s0.T, np.dot(C, e_s0))
    I4n = np.dot(e_n0.T, np.dot(C, e_n0))
    I5f = np.dot(e_f0.T, np.dot(np.linalg.matrix_power(C, 2), e_f0))
    I5s = np.dot(e_s0.T, np.dot(np.linalg.matrix_power(C, 2), e_s0))
    I5n = np.dot(e_n0.T, np.dot(np.linalg.matrix_power(C, 2), e_n0))

    # calculate the coupling invariants I8 (note: I8ij = I8ji)
    I8fs = np.dot(e_f0.T, np.dot(C, e_s0))
    I8fn = np.dot(e_f0.T, np.dot(C, e_n0))
    I8ns = np.dot(e_n0.T, np.dot(C, e_s0))

    return I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns


def calculate_sigma_sym(psi, num_params):
    """
    Calculates the symbolic stress tensor sigma given a symbolic strain energy function psi.
    :param psi: strain energy function
    :param num_params: number of parameters in psi
    :return: stress tensor sigma
    """
    dim = 3  # dimension of the tensors

    # symbolic representation of the deformation gradient F, the invariants and the parameters
    global F_sym, p_sym, I1_sym, I2_sym, I3_sym, I4f_sym, I4s_sym, I4n_sym, I5f_sym, I5s_sym, I5n_sym, I8fs_sym, I8fn_sym, I8ns_sym

    F_sym = sp.MatrixSymbol("F_sym", dim, dim)
    I1_sym, I2_sym, I3_sym, I4f_sym, I4s_sym, I4n_sym, I5f_sym, I5s_sym, I5n_sym, I8fs_sym, I8fn_sym, I8ns_sym = sp.symbols(
        'I1_sym I2_sym I3_sym I4f_sym I4s_sym I4n_sym I5f_sym I5s_sym I5n_sym I8fs_sym I8fn_sym I8sn_sym', real=True)
    I_sym_lst = [I1_sym, I2_sym, I3_sym, I4f_sym, I4s_sym, I4n_sym, I5f_sym, I5s_sym, I5n_sym, I8fs_sym, I8fn_sym,
                 I8ns_sym]

    p_sym = [sp.symbols("p%i_sym" % (n)) for n in range(num_params)]

    # calculate the symbolic representations of the functions used to calculate the invariants depending on F (for the off-diagonal elements)
    C_sym = F_sym.T * F_sym
    I1_func, I2_func, I3_func, I4f_func, I4s_func, I4n_func, I5f_func, I5s_func, I5n_func, I8fs_func, I8fn_func, I8ns_func = invariants(
        C_sym)
    I_func_lst = [I1_func, I2_func, I3_func, I4f_func, I4s_func, I4n_func, I5f_func, I5s_func, I5n_func, I8fs_func,
                  I8fn_func, I8ns_func]

    # call strain energy function
    psi = psi(I1_sym, I2_sym, I3_sym, I4f_sym, I4s_sym, I4n_sym, I5f_sym, I5s_sym, I5n_sym, I8fs_sym, I8fn_sym,
              I8ns_sym, p_sym)

    # calculate the components of the stress tensor
    sigma_sym = sp.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            # sigma_ij = sum_k F_ik dpsi/dF_jk = sum_k F_ik sum_i dpsi/dI_i dI_i/dF_jk
            sum_k = 0
            for k in range(dim):
                sum_i = 0
                for I_i_sym, I_i_func in zip(I_sym_lst, I_func_lst):
                    sum_i += sp.diff(psi, I_i_sym) * sp.diff(I_i_func, F_sym[j, k])
                sum_k += F_sym[i, k] * sum_i
            sigma_sym[i, j] = sum_k

    # incorporate the incompressibility conditions
    sigma_sym[0, 0] = sigma_sym[0, 0] - sigma_sym[2, 2]
    sigma_sym[1, 1] = sigma_sym[1, 1] - sigma_sym[2, 2]

    return sigma_sym


def calculate_stress_tensor_function(psi, num_params):
    """
    Converts the stress tensor sigma to a function of F, p and I.
    :param psi: strain energy function
    :param num_params: number of parameters in the strain energy function
    :return: sigma_function, i.e. a function sigma(F,p1,...,pn, I1,...,I8ns)
    """
    dim = 3
    axes_to_indices = {"f": 0, "s": 1, "n": 2}  # convert string labelling of axes to indices

    sigma_sym = calculate_sigma_sym(psi, num_params)

    # convert the symbolic function to a numpy function to save computation time. Therefore, we need to tell the
    # lambdify function all the variables in the function
    symbols_lst = [I1_sym, I2_sym, I3_sym, I4f_sym, I4s_sym, I4n_sym, I5f_sym, I5s_sym, I5n_sym, I8fs_sym, I8fn_sym,
                   I8ns_sym]
    symbols_lst.extend(p_sym)
    gamma_sym = sp.symbols("gamma_sym", real=True)
    symbols_lst.append(gamma_sym)

    sigma_function = {}
    for shear in ['fs', 'fn', 'sf', 'sn', 'nf', 'ns']:
        sigma_sym_shear = copy.copy(sigma_sym)
        ind1 = axes_to_indices[shear[0]]
        ind2 = axes_to_indices[shear[1]]

        # already put the deformation gradient in the expression because of all the null components this saves
        # computation time
        F = deformation_gradient(shear, gamma_sym)
        sigma_sym_shear = sigma_sym_shear.subs(F_sym, F)
        for i in range(dim):
            for j in range(dim):
                sigma_sym_shear = sigma_sym_shear.subs(F_sym[i, j], F[i, j])
        sigma_sym_shear = sp.simplify(sigma_sym_shear[ind1, ind2])

        # convert to numpy
        sigma_shear = sp.lambdify(symbols_lst, sigma_sym_shear, modules=[{"Trace": np.trace}, "numpy"])
        sigma_function[shear] = sigma_shear

    return sigma_function


def evaluate_stress_tensor_function(shear, gamma, p, sigma_function):
    """
    Evaluates the stress tensor sigma.
    :param shear: type of shear ('fs', 'sf', 'sn', 'ns', 'fn' or 'nf')
    :param gamma: amount of shear
    :param p: list containing all parameters of the strain energy function
    :param sigma_function: function defining sigma as sigma(F,p1,...,pn, I1,...,I8ns)
    :return: sigma_ij, i.e. the matrix element of sigma corresponding to the given type of shear
    """

    # calculate real deformation gradient and corresponding invariants for the given type of shear with numpy
    F = deformation_gradient_np(shear, gamma)
    I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns = invariants_np(F)

    # evaluate the sigma function with the given parameters
    sigma_function_comp = sigma_function[shear]
    sigma = sigma_function_comp(float(I1), float(I2), float(I3), float(I4f), float(I4s), float(I4n), float(I5f), float(I5s),
                           float(I5n), float(I8fs), float(I8fn), float(I8ns), *p, gamma)

    # return the component of the sigma tensor belonging to the given type of shear
    return float(sigma)


def calculate_S_tensor_function(psi, num_params):
    """
    Calculates the second Piola-Kirchhoff stress tensor sigma as as symbolic function of F, p and Ir.
    :param psi: strain energy function
    :param num_params: number of parameters in the strain energy function
    :return: S function, i.e. a function S(F,p1,...,pn, I1,...,I8ns)
    """
    axes_to_indices = {"f": 0, "s": 1, "n": 2}  # convert string labelling of axes to indices
    dim = 3

    lam_f_sym = sp.symbols("lam_f_sym", real=True)
    lam_s_sym = sp.symbols("lam_s_sym", real=True)

    F_sym_lam = sp.Matrix([[lam_f_sym, 0, 0], [0, lam_s_sym, 0], [0, 0, 1 / (lam_f_sym * lam_s_sym)]])
    sigma_sym = calculate_sigma_sym(psi, num_params)
    S_sym = F_sym_lam.det() * F_sym_lam.inv() * sigma_sym * F_sym_lam.inv().adjugate()

    # convert the symbolic function to a numpy function to save computation time. Therefore, we need to tell the
    # lambdify function all the variables in the function
    symbols_lst = [I1_sym, I2_sym, I3_sym, I4f_sym, I4s_sym, I4n_sym, I5f_sym, I5s_sym, I5n_sym, I8fs_sym, I8fn_sym,
                   I8ns_sym]
    symbols_lst.extend(p_sym)
    symbols_lst.append(lam_f_sym)
    symbols_lst.append(lam_s_sym)

    S_function = {}
    for shear in ['ff', 'ss']:
        ind1 = axes_to_indices[shear[0]]
        ind2 = axes_to_indices[shear[1]]
        S_sym_shear = copy.copy(S_sym[ind1, ind2])

        F = np.array([[lam_f_sym, 0, 0], [0, lam_s_sym, 0], [0, 0, 1 / (lam_f_sym * lam_s_sym)]])

        # already put the deformation gradient in the expression because of all the null components this saves
        # computation time
        S_sym_shear = S_sym_shear.subs(F_sym, F)
        S_sym_shear = S_sym_shear.subs(F_sym_lam, F)
        for i in range(dim):
            for j in range(dim):
                S_sym_shear = S_sym_shear.subs(F_sym[i, j], F[i, j])
        S_sym_shear = sp.simplify(S_sym_shear)

        # convert to numpy
        S_shear = sp.lambdify(symbols_lst, S_sym_shear, modules=[{"Trace": np.trace}, "numpy"])
        S_function[shear] = S_shear

    return S_function


def evaluate_S_tensor_function(shear, lam_f, lam_s, p, S_function):
    """
    Evaluates the Piola-Kirchhoff stress tensor S.
    :param shear: type of shear ('ff', 'ss')
    :param lam_f, lam_s: amount of shear in f or respectively s direction
    :param p: list containing all parameters of the strain energy function
    :param S_function: function defining S as S(F,p1,...,pn, I1,...,I8ns)
    :return: S_ij, i.e. the matrix element of S corresponding to the given type of shear
    """

    # calculate real deformation gradient and corresponding invariants for the given type of shear with numpy
    F = np.array([[lam_f, 0, 0], [0, lam_s, 0], [0, 0, 1 / (lam_f * lam_s)]])
    I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns = invariants_np(F)


    # evaluate the sigma function with the given parameters
    S_function_comp = S_function[shear]
    S = S_function_comp(float(I1), float(I2), float(I3), float(I4f), float(I4s), float(I4n), float(I5f), float(I5s),
                           float(I5n), float(I8fs), float(I8fn), float(I8ns), *p, lam_f, lam_s)

    # return the component of the sigma tensor belonging to the given type of shear
    return float(S)


def calculate_stress_tensor_function_diagonal(psi, num_params):
    """
    Calculates diagonal elements of the Cauchy stress tensor sigma.
    :param psi: strain-energy-function, i.e. a function with parameters I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns
    :param num_params: number of parameters in the strain energy function
    :return: sigma_function, i.e. a symbolic function sigma(F,p1,...,pn, I1,...,I8ns)
    """
    dim = 3
    axes_to_indices = {"f": 0, "s": 1, "n": 2}  # convert string labelling of axes to indices

    sigma_sym = calculate_sigma_sym(psi, num_params)

    # convert the symbolic function to a numpy function to save computation time. Therefore, we need to tell the
    # lambdify function all the variables in the function
    symbols_lst = [I1_sym, I2_sym, I3_sym, I4f_sym, I4s_sym, I4n_sym, I5f_sym, I5s_sym, I5n_sym, I8fs_sym, I8fn_sym,
                   I8ns_sym]
    symbols_lst.extend(p_sym)
    lam_f_sym = sp.symbols("lam_f_sym", real=True)
    lam_s_sym = sp.symbols("lam_s_sym", real=True)
    symbols_lst.append(lam_f_sym)
    symbols_lst.append(lam_s_sym)

    sigma_function = {}
    for shear in ['ff', 'ss']:
        sigma_sym_shear = copy.copy(sigma_sym)
        ind1 = axes_to_indices[shear[0]]
        ind2 = axes_to_indices[shear[1]]

        # already put the deformation gradient in the expression because of all the null components this saves
        # computation time
        F = sp.Matrix([[lam_f_sym, 0, 0], [0, lam_s_sym, 0], [0, 0, 1 / (lam_f_sym * lam_s_sym)]])

        sigma_sym_shear = sigma_sym_shear.subs(F_sym, F)
        for i in range(dim):
            for j in range(dim):
                sigma_sym_shear = sigma_sym_shear.subs(F_sym[i, j], F[i, j])
        sigma_sym_shear = sp.simplify(sigma_sym_shear[ind1, ind2])

        # convert to numpy
        sigma_shear = sp.lambdify(symbols_lst, sigma_sym_shear, modules=[{"Trace": np.trace}, "numpy"])
        sigma_function[shear] = sigma_shear

    return sigma_function


def evaluate_stress_tensor_function_diagonal(shear, lam_f, lam_s, p, sigma_function):
    """
    Evaluates the diagonal elements of the stress tensor function.
    :param shear: type of shear ('ff', 'ss')
    :param gamma: amount of shear
    :param p: list containing all parameters of the strain energy function
    :param sigma_function: function defining sigma as sigma(F,p1,...,pn, I1,...,I8ns)
    :return: sigma_ii, i.e. the matrix element of sigma corresponding to the given type of shear
    """

    # calculate real deformation gradient and corresponding invariants for the given type of shear with numpy
    F = np.array([[lam_f, 0, 0], [0, lam_s, 0], [0, 0, 1 / (lam_f * lam_s)]])
    I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns = invariants_np(F)


    # evaluate the sigma function with the given parameters
    sigma_function_comp = sigma_function[shear]
    sigma = sigma_function_comp(float(I1), float(I2), float(I3), float(I4f), float(I4s), float(I4n), float(I5f), float(I5s),
                           float(I5n), float(I8fs), float(I8fn), float(I8ns), *p, lam_f, lam_s)

    # return the component of the sigma tensor belonging to the given type of shear
    return float(sigma)