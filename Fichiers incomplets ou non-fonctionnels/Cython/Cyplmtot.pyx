import numpy as np
cimport numpy as np
from cython.parallel cimport prange

def plm(int nb_c, float x_sump, float[:,:] S, float[:,:] lada, float[:,:] ds, float[:,:] A,float[:,:] b):
    nogil = True
    cpdef float[:,:] xx, xi, x_c_old, x, temp_value,s, rumba, ramda, q, kk1, filterP, filter2,filter1, fd_v,dss,change_factor_vector, add_sd_vv, add_fd_v, active_id_nb, active_id, active_change, Pt, P, K 

    cpdef int  nb_activ, count_offense, count_offense1, count_iter, change_factor, j, id_lambda, o,I
    
    with nogil:
        # filter indices
        active_id_nb = np.array([range(0, nb_c + 1)])
    
        # initialize x
        x = np.ones((1, nb_c)) / (2 * nb_c)
    
        # constraints
    
        # initialize the filter
        filter1 = np.zeros((nb_c + 1, 1)) == 1
        filter1 = filter1.T
    
        # For while loop initialization
    
        count_iter = 0
    
        x_c_old = np.ones((1, nb_c)) * np.inf
    
        active_change = np.zeros((1, nb_c)) == 1
    
        verify_one = True
    
        while np.sum((x - x_c_old) ** 2) > 1E-6: #ok
    
            x_c_old = x
            cont = True
    
            while cont: #ok
                # filter
                filter1[:, -1] = 1  # sum of all components
    
                if count_iter == 50: #ok
                    break
    
                active_id = active_id_nb[filter1]
    
                # modify x if there are 0 values
                # avoid division by zero
                xi = x
                xi[xi < 0] = 0
    
                # calculate the current spectrum and the first derivative
                s = x_sump * np.sum(S * xi.T, axis=0)
                s = np.array([s])
    
                # calculate de Jacobian vector and the Hessian matrix
    
                lada = np.array(lada)
                filter2 = lada > 0 # filter for lambda positive
                
    
    
    
                
    
    
                # calculate the Jacobian vector
                fd_v = -ds
                add_fd_v = ds*(lada/s)
                fd_v[:,filter2[0]] = fd_v[:,filter2[0]] + add_fd_v[:, filter2[0]]
                # only add values with lambda
                # Jacobian vector
    
                q = np.sum(fd_v, axis=1).T
    
                # calculate Hessian matrix
                kk1 = -lada * s ** (-2)
                dss = ds * kk1
                P = np.zeros((nb_c, nb_c))
    
                for j in range(0, nb_c): #ok
                    add_sd_vv = dss[j:, filter2] * ds[j, filter2]
                    P[j, j:] = np.sum(add_sd_vv, axis=1)
                    P[j + 1:, j] = P[j, j + 1:].T
                
                
                # Combine  the Hessian matrix with the activated constraints
                nb_activ = len(active_id)  # calculate number of activated constraints
    
                K = np.append( np.append(P, A[filter1[0, :].T, :].T, axis=1), np.append(A[filter1[0, :].T, :], np.zeros((nb_activ, nb_activ)), axis=1), axis=0)
    
                # value verification
                # if an error occured, "inf" values will have appeared. if "inf", stop the iteration
                Pt = P.reshape(np.prod(np.shape(P)),1)
                filterP = np.isinf(Pt)
                if Pt[filterP].size == 0:
    
    
                    # matrix opertation
                    # Calculate delta_x and lagrange multiplier (both in xx)
    
    
    
                    xx = np.linalg.lstsq(K,np.append(-q, b[filter1.T], axis=0))[0]
    
                    change_factor = 1
                    # change delta x
                    xx[0:nb_c] = change_factor * xx[0:nb_c]
    
                    # constraints check
                    # calculate temporary values
                    # if all constraints are respected, these values will be the new ones
    
                    temp_value = x + xx[0:nb_c].T
    
                    count_offense = 0
                    count_offense1 = 0
    
                    # Is xj now negative?
    
                    change_factor_vector = np.ones((1, nb_c + 1))
                    for j in np.arange(0, nb_c):
    
                        if temp_value[:, j] < -.0001 and active_change[:, j] == 0:
                            count_offense = count_offense + 1
                            count_offense1 = 1
                            change_factor_vector[0, j] = -x[0, j] / xx[j]
    
                            
                    if count_offense1 == 1:
                        change_factor = np.min(change_factor_vector)
                        I = np.argmin(change_factor_vector)
                        filter1[:, I] = True
                        b[I, :] = 0 + x[:, I]
                        active_change[:, I] = 1
    
    
                    else: # ok
                        x = x + xx[0:nb_c].T
                        x = x / np.sum(x)
                        b = np.zeros((nb_c + 1, 1))
                        active_change = np.zeros((1, nb_c)) == 1
                        cont = False
    
    
                    # All constraints are respected
                    # Verify if the lambda are positive
                    # If yes, erase the constraint
                    if count_offense == 0 and len(xx) > nb_c:
                        for id_lambda in range(len(xx) - nb_c-1,-1,-1):
                            if xx[id_lambda + nb_c] > 0: #ok
                                filter1[:,active_id[id_lambda]] = False
    
    
    
                    # P values with "inf" values
    
                else:
                    c = np.max(x)
                    o = np.argmax(x)
                    filter1[:,:-1] = False
                    x = x + 0.05 / (nb_c - 1)
                    x[:,o] = c - 0.05
                    cont = False
    
                if (np.sum(x[x > 0.99], axis=0) > 0.99) and verify_one:
                    verify_one = False
                    c = np.max(x)
                    o = np.argmax(x)
                    filter1[:,1: -1] = False
                    x = x + 0.05 / (nb_c - 1)
                    x[:,o] = c - 0.05
                    cont = False
    
                    
                count_iter = count_iter + 1
        xshape = x.shape
        lenx = xshape[0]*xshape[1]
        x = x.reshape(1,lenx)
        for ip in range(lenx):
            if x[0,ip] < 0:
                x[0,ip] = 0
        x = x.reshape(xshape[0],xshape[1])
    return x



def C_plm(np.ndarray x, np.ndarray s, np.ndarray lada, np.ndarray x_sum, int nb_c):
    
    cpdef float[:,:] A, b, C,S,S2,C_sum1, C_sum
    cpdef int n, m, ids
    A = np.array(np.append(-np.eye(nb_c), np.ones((1, nb_c)), axis=0))
    b = np.array([np.append(np.zeros((nb_c, 1)), 1)]).T

    # initialize C

    [n, m] = np.shape(x)
    C = np.zeros((n, nb_c))
        
    # FOR ALL PIXELS
    # Paralled FOR loop

    for ids in prange(n, nogil=True):
        C[ids, :] = plm(nb_c, x_sum[ids, 0], s, lada[ids, :], x_sum[ids, 0] * s, A, b)
        # C[ids,:] = cls.plm(nb_c,x_sum[ids,0],s,lada[ids,:],x_sum[ids,0]*s,A,b)
        


    # avoid errors (this part should not be necessary)

    C[np.isnan(C)] = 1 / nb_c
    C[np.isinf(C)] = 1 / nb_c
    C[C < 0] = 0
    C_sum1 = np.array([np.sum(C, axis=1)])
    C_sum = C_sum1.T @ np.ones((1, np.size(C, axis=1)))
    C = C / C_sum



    S2 = np.linalg.inv(C.T @ C) @ C.T @ x  # Multilinear regression
    S2[S2 <= np.spacing(1)] = np.spacing(1)  # avoid 0 values
    s = S2 / (np.sum(S2, axis=1) * np.ones((np.size(S2, axis=1), 1))).T  # Normalization


    return C,s