import jax
import jax.numpy as jnp


def saturated_pressure(T):
    # saturated pressure in Pa of water in air at given temperature   
    # using Antoine equation
    # https://en.wikipedia.org/wiki/Vapour_pressure_of_water#:~:text=The%20saturation%20vapor%20pressure%20is,it%20would%20evaporate%20or%20sublimate.
    Tc = T - 273
    return jnp.where(Tc < 99, 10**(8.07131 - 1730.63/(233.426 + Tc))*133.322, 10**(8.14019 - 1810.94/(244.485 + Tc))*133.322)
    
def moisture_content(T, density_air, relative_humidity = 1):
    # calculates the equilibrium moisture content in the air at given temperature
    # assumes ideal gas law
    mw_air = 28.96e-3 # kg/mol molecular weight of air
    mw_water = 18.01e-3 # kg/mol molecular weight of water 
    P_sat = saturated_pressure(T) # Pa
    return vapor_pressure_to_moisture(P_sat, T, density_air, relative_humidity)

def vapor_pressure_to_moisture(vp, T, density_air, relative_humidity = 1):
    # converts vapor pressure to moisture content
    # we use ideal gas law
    mw_water = 18.01e-3 # kg/mol molecular weight of water 
    water_concentration = vp/(8.314*T)
    return water_concentration*mw_water*relative_humidity/density_air

def _density_air(X, T):
    # X is moisture content is dry basis
    # T is temperature in K
    # Assuming total pressure to be 1 atm
    mu = 1.606 # mw air/ mw water
    xa = 1/(1 + X*mu)
    return 1e5/(8.314*T)*(xa*28.96 + (1 - xa)*18.01)*1e-3

def _mass_transfer_coefficient(x, B, c):
    # negative value is eliminated in the final equation using exp

    X, Ts, Y, Ta, Ye = x
    
    if isinstance(B, dict):

        x =  jnp.append(jnp.append(X, jnp.append(Ts, jnp.append(Ta, Y))), c.density_product)
        n = len(B["weight"])
        for i, (weight, bias) in enumerate(zip(B["weight"], B["bias"])):
            x = jnp.dot(weight, x) + bias
            if i < n - 1:
                x = jnp.tanh(x)

        return jnp.exp(x)
    else: return B

def _heat_transfer_coefficient(x, A, c):
    # negative value is eliminated in the final equation using exp
    
    X, Ts, Y, Ta, Ye = x
    
    if isinstance(A, dict):
        x =  jnp.append(jnp.append(X, jnp.append(Ts, jnp.append(Ta, Y))), c.density_product)
        n = len(A["weight"])
        for i, (weight, bias) in enumerate(zip(A["weight"], A["bias"])):
            x = jnp.dot(weight, x) + bias
            if i < n - 1 :
                x = jnp.tanh(x) 
        
        return jnp.exp(x)
    else: return A