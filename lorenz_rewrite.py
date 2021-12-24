import numpy as np
dt, warmup, traintime, testtime = 0.05, 5., 20., 45.; maxtime = warmup+traintime+testtime

from scipy.integrate import solve_ivp
def lorenz(t,y):
    sigma, beta, rho = 10, 8/3., 28
    return sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]  
evaluation_time_interval=np.linspace(0,maxtime,round(maxtime/dt)+1)    
lorenz_solution = solve_ivp(lorenz, (0,maxtime),
     [17.67715816276679, 12.931379185960404, 43.91404334248268],
     t_eval=evaluation_time_interval, method='RK45') 
warmup_pts, traintime_pts, testtime_pts, maxtime_pts = \
round(warmup/dt), round(traintime/dt), round(testtime/dt), lorenz_solution.y.shape[1]
warmtrain_pts=warmup_pts+traintime_pts
del solve_ivp, evaluation_time_interval, maxtime, traintime, testtime, warmup, dt

N, delay_taps = lorenz_solution.y.shape[0], 4; skip_tap_to_tap = delay_taps + 1
size_linear, size_nonlinear = delay_taps*(N-1), int(N*(N+1)/2)
size_total = size_linear + size_nonlinear
assert delay_taps*skip_tap_to_tap <= warmup_pts, "not enough warmup_pts!"
        
x = np.zeros((size_linear,maxtime_pts))
#fill linear part
ij_ixs = [(delay, j) for delay in range(delay_taps) for j in range(delay,maxtime_pts)]
for delay, j in ij_ixs:
    #only include x and y
    x[(N-1)*delay:(N-1)*(delay+1),j]=lorenz_solution.y[:2,j-delay*skip_tap_to_tap]
out = np.ones((size_total+1,maxtime_pts-warmup_pts))
post_warmup_x = x[:,warmup_pts:maxtime_pts]; out[1:size_linear+1,:] = post_warmup_x
#fill nonlinear part
count, ij_ixs = 0, [(i,j) for i in range(N) for j in range(i,N)]
for i,j in ij_ixs:
    out[size_linear+1+count,:]=post_warmup_x[i,:]*post_warmup_x[j,:]; count+=1
del x
        
#ridge regression: train W_out
ridge_param = .05
W_out = lorenz_solution.y[2,warmup_pts:warmtrain_pts] @ out[:,0:traintime_pts].T \
        @ np.linalg.pinv( out[:,0:traintime_pts] @ out[:,0:traintime_pts].T \
                          + ridge_param*np.identity(size_total+1) ) 

z_predict = W_out @ out
