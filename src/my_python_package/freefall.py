import numpy as np

def ode_freefall_euler(g0, dg_dz, cd_star, H, dt):
   
    #Solve free fall ODE using Euler's method.
    
    #parameters:
    
    #g0 : float, gravity at surface 
    #dg_dz : float, free-air gradient of gravity 
    #cd_star : float, mass-normalized drag coefficient (1/s)
    #H : float, Drop height 
    #dt : float, initial timestep (s)
    
    #Returns
   
   # t, z, v : np.ndarray, time, displacement, v
    
    
    # initialize lists
    t, z, v = [0.0], [0.0], [0.0]
    
    while z[-1] < H:
        # a from ODE: dv/dt
        a = g0 + dg_dz * z[-1] - cd_star * v[-1]
        
        # Euler update
        v_new = v[-1] + dt * a
        z_new = z[-1] + dt * v[-1]
        t_new = t[-1] + dt
        
        # adjust for final step in case overshooting H

        if z_new > H:
            # linear correction of dt
            dt_last = (H - z[-1]) / v[-1] if v[-1] > 0 else dt
            v_new = v[-1] + dt_last * a
            z_new = H
            t_new = t[-1] + dt_last
        
        
        v.append(v_new)
        z.append(z_new)
        t.append(t_new)
    
    return np.array(t), np.array(z), np.array(v)


def ode_freefall_rk4(g0, dg_dz, cd_star, H, dt):
    
    
    t, z, v = [0.0], [0.0], [0.0]
    
    def f(z, v):

        #return derivatives#

        dzdt = v
        dvdt = g0 + dg_dz * z - cd_star * v
        return dzdt, dvdt
    
    while z[-1] < H:
        z_n, v_n = z[-1], v[-1]
        
        # RK4 steps
        k1z, k1v = f(z_n, v_n)
        k2z, k2v = f(z_n + 0.5*dt*k1z, v_n + 0.5*dt*k1v)
        k3z, k3v = f(z_n + 0.5*dt*k2z, v_n + 0.5*dt*k2v)
        k4z, k4v = f(z_n + dt*k3z, v_n + dt*k3v)
        
        z_new = z_n + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
        v_new = v_n + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        t_new = t[-1] + dt
        
        # adjust final step
        if z_new > H:
            dt_last = (H - z[-1]) / v[-1] if v[-1] > 0 else dt
            k1z, k1v = f(z_n, v_n)
            k2z, k2v = f(z_n + 0.5*dt_last*k1z, v_n + 0.5*dt_last*k1v)
            k3z, k3v = f(z_n + 0.5*dt_last*k2z, v_n + 0.5*dt_last*k2v)
            k4z, k4v = f(z_n + dt_last*k3z, v_n + dt_last*k3v)
            
            z_new = z_n + (dt_last/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
            v_new = v_n + (dt_last/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
            t_new = t[-1] + dt_last
            z_new = H
        
        
        v.append(v_new)
        z.append(z_new)
        t.append(t_new)
    
    return np.array(t), np.array(z), np.array(v)


def compute_cd_star(diameter=0.015, rho=7800, mu=1.827e-5):
   
   # compute mass-normalized drag coefficient cD* for a sphere in air.
    
   # parameters
    
  #  diameter : float, sphere diameter 
   # rho : float density 
  #  mu : float viscosity of air 
   # 
   # Returns
    
  #  cd_star : float, mass-normalized drag coefficient 
    
    r = diameter / 2.0
    volume = (4.0/3.0) * np.pi * r**3
    mass = rho * volume
    cD = 6.0 * np.pi * mu * r
    return cD / mass