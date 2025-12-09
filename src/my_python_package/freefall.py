import numpy as np

def ode_freefall_euler(g0, dg_dz, cd_star, H, dt):
    """
    #Solve free fall ODE using Euler's method.
    
    #Parameters:
    ----------
    #g0 : float, Gravity at surface (m/s^2)
    #dg_dz : float
        Free-air gradient of gravity ((m/s^2)/m)
    #cd_star : float
        Mass-normalized drag coefficient (1/s)
    #H : float
        Drop height (m)
    #dt : float
        Initial timestep (s)
    
    #Returns
   # -------
   # t, z, v : np.ndarray
    #    Arrays of time, displacement, velocity
    """
    # Initialize lists
    t, z, v = [0.0], [0.0], [0.0]
    
    while z[-1] < H:
        # Acceleration from ODE: dv/dt
        a = g0 + dg_dz * z[-1] - cd_star * v[-1]
        
        # Euler update
        v_new = v[-1] + dt * a
        z_new = z[-1] + dt * v[-1]
        t_new = t[-1] + dt
        
        # Adjust final step if overshooting H
        if z_new > H:
            # linear correction of dt
            dt_last = (H - z[-1]) / v[-1] if v[-1] > 0 else dt
            v_new = v[-1] + dt_last * a
            z_new = H
            t_new = t[-1] + dt_last
        
        # Append
        v.append(v_new)
        z.append(z_new)
        t.append(t_new)
    
    return np.array(t), np.array(z), np.array(v)


def ode_freefall_rk4(g0, dg_dz, cd_star, H, dt):
    """
    Solve free fall ODE using classical RK4 method.
    """
    # Initialize lists
    t, z, v = [0.0], [0.0], [0.0]
    
    def f(z, v):
        """Return dz/dt, dv/dt"""
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
        
        # Adjust final step if overshooting H
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
        
        # Append
        v.append(v_new)
        z.append(z_new)
        t.append(t_new)
    
    return np.array(t), np.array(z), np.array(v)


def compute_cd_star(diameter=0.015, rho=7800, mu=1.827e-5):
    """
    Compute mass-normalized drag coefficient cD* for a sphere in air.
    
    Parameters
    ----------
    diameter : float
        Sphere diameter (m)
    rho : float
        Density of sphere material (kg/m^3)
    mu : float
        Dynamic viscosity of air (kg/(m·s))
    
    Returns
    -------
    cd_star : float
        Mass-normalized drag coefficient (1/s)
    """
    r = diameter / 2.0
    volume = (4.0/3.0) * np.pi * r**3
    mass = rho * volume
    cD = 6.0 * np.pi * mu * r
    return cD / mass