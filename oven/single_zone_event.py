from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .ode_event_rk_jax import odeint_event, odeint_rk_grid
from .data_structures import Constants
from .utils import root_finding_rev, root_finding_fwd


@partial(jax.jit, static_argnums = (3, )) # pass namedtuple for funtion and not constant dict
def oven_zone(xinit : jnp.array, parameters : dict, controls : dict, constant : Constants, first_zone : bool = True, recirculation : bool = False):

    # merge dictionaries
    parameters = {**parameters, **controls}

    # model constants
    radius_particle = constant.radius_particle # m radius of particle
    density_particle = constant.density_particle # kg/m3 density of particle
    voidage = constant.voidage # 1 - density_bulk/density_particle
    area_volume_ratio = 2 / radius_particle # this changes depending on the particle

    enthalpy_vaporization_water = constant.enthalpy_vaporization_water # J/kg
    specific_heat_capacity_air = constant.specific_heat_capacity_air # J/kg/K
    specific_heat_capacity_solid = constant.specific_heat_capacity_solid # J/kg/K
   
    ny = constant.ny
    product_height = constant.product_height # m
    diff_elem_y = product_height / ny

    equilibrium_moisture = constant.equilibrium_moisture # kg water / kg dry solid

    # scaling parameters
    moisture_max = constant.moisture_max
    moisture_min = constant.moisture_min
    temperature_max = constant.temperature_max
    temperature_min = constant.temperature_min

    # heat and mass transfer functions. Should take arguments as ((X, Ts, Y, Ta, Y_equill), parameters, **kwargs)
    scaled_moisture_content  = constant.scaled_moisture_content
    scaled_mass_transfer_coefficient = constant.scaled_mtc
    scaled_heat_transfer_coefficient = constant.scaled_htc
    scaled_reaction_rate = constant.scaled_k
    density_air = constant.density_air # kg/m3 density of air

    kwargs = constant.ode_kwargs._asdict()
    reuse_inverse = kwargs.pop("reuse")


    def event(x, t):
        """
        Event defines the switch from constant rate drying to falling rate drying
        """
        return jnp.array([x[0] - x[-1]])


    def solid_temperature_jump(xi, p):
        """
        Models the initial jump in the temperature during the constant rate drying
        """

        def func(T, p):
            
            p, ms = p
            ma, ta = p["init_moisture_air"], p["init_temperature_air"]
            _mtc, _htc = p["mass_transfer_coefficient_constant"], p["heat_transfer_coefficient_constant"] # scale mass transfer coefficient

            den_air = density_air(ma, ta)
            ye = scaled_moisture_content(T, den_air)
            mtc = scaled_mass_transfer_coefficient((ms, T, ma, ta, ye), _mtc, constant)
            htc = scaled_heat_transfer_coefficient((ms, T, ma, ta, ye), _htc, constant)
            # constant_jump is added to prevent the system exploding/stiff
            return (ta - p["constant_jump"] * mtc * den_air * (ye - ma) / htc / (temperature_max - temperature_min)).reshape(T.shape)
       
        Topt = jax.lax.cond(first_zone & (event(xi, 0)[0] > 0),
                            lambda : root_finding_fwd(lambda T, p : T - func(T, p), xi[1], (p, xi[0]), reuse_inverse = reuse_inverse),
                            lambda : xi[1])    
        return Topt


    def air_model_falling_rate(y, t, p):
            x, p = p
            moisture_air, temperature_air = y
            moisture_solid, temperature_solid, _, critical_moisture = x
            _velocity = p["init_velocity_air"]
            heat_transfer_coefficient_falling = p["heat_transfer_coefficient_falling"]
            mass_transfer_coefficient_falling = p["mass_transfer_coefficient_falling"]

            den_air = density_air(moisture_air, temperature_air)
            y_equil = scaled_moisture_content(temperature_solid, den_air)
            nn_inputs = (moisture_solid, temperature_solid, moisture_air, temperature_air, y_equil)

            htc = scaled_heat_transfer_coefficient(nn_inputs, heat_transfer_coefficient_falling, constant)
            mtc = scaled_mass_transfer_coefficient(nn_inputs, mass_transfer_coefficient_falling, constant) * ((moisture_solid - equilibrium_moisture) / (critical_moisture - equilibrium_moisture))

            dmoisture = mtc * (y_equil - moisture_air) / _velocity / enthalpy_vaporization_water
            dtemperature = (- htc * (temperature_air - temperature_solid) + mtc * den_air * (y_equil - moisture_air) * (moisture_max - moisture_min) / (temperature_max - temperature_min)) / (_velocity * specific_heat_capacity_air * den_air)

            return jnp.concatenate((
                (1 - voidage) * area_volume_ratio * dmoisture,
                (1 - voidage) * area_volume_ratio * dtemperature * p["constant"]
            ))

    def air_model_constant_rate(y, t, p):
        x, p = p
        moisture_air, temperature_air = y
        moisture_solid, temperature_solid, _, _ = x
        _velocity = p["init_velocity_air"]
        heat_transfer_coefficient_falling = p["heat_transfer_coefficient_constant"]
        mass_transfer_coefficient_falling = p["mass_transfer_coefficient_constant"]

        den_air = density_air(moisture_air, temperature_air)
        y_equil = scaled_moisture_content(temperature_solid, den_air)
        nn_inputs = (moisture_solid, temperature_solid, moisture_air, temperature_air, y_equil)

        htc = scaled_heat_transfer_coefficient(nn_inputs, heat_transfer_coefficient_falling, constant)
        mtc = scaled_mass_transfer_coefficient(nn_inputs, mass_transfer_coefficient_falling, constant)

        dmoisture = mtc * (y_equil - moisture_air) / _velocity / enthalpy_vaporization_water
        dtemperature = (- htc * (temperature_air - temperature_solid) + mtc * den_air * (y_equil - moisture_air) * (moisture_max - moisture_min) / (temperature_max - temperature_min)) / (_velocity * specific_heat_capacity_air * den_air)

        return jnp.concatenate((
            (1 - voidage) * area_volume_ratio * dmoisture,
            (1 - voidage) * area_volume_ratio * dtemperature * p["constant"]
        ))

    def solid_model_falling_rate(x, t, p):
        """
        Falling rate drying equations of solid (Single trajectory)
        """
        air, p = p
        moisture_air, temperature_air = air
        moisture_solid, temperature_solid, cure, critical_moisture = x # mositure in particle, temperature of solid
        mass_transfer_coefficient = p["mass_transfer_coefficient_falling"]
        heat_transfer_coefficient = p["heat_transfer_coefficient_falling"]
        reaction_rate = p["reaction_rate"]
        activation_energy = p["activation_energy"]
       
        den_air = density_air(moisture_air, temperature_air)
        y_equil = scaled_moisture_content(temperature_solid, den_air)

        mtc = scaled_mass_transfer_coefficient((moisture_solid, temperature_solid, moisture_air, temperature_air, y_equil), mass_transfer_coefficient, constant)
        htc = scaled_heat_transfer_coefficient((moisture_solid, temperature_solid, moisture_air, temperature_air, y_equil), heat_transfer_coefficient, constant)
        mtr = mtc * den_air * ((moisture_solid - equilibrium_moisture)/(critical_moisture - equilibrium_moisture)) * (y_equil - moisture_air)
       
        dmoisture = - area_volume_ratio * mtr / enthalpy_vaporization_water / density_particle
        dtemperature = (
            area_volume_ratio * (htc * (temperature_air - temperature_solid) - mtr * (moisture_max - moisture_min)/(temperature_max - temperature_min))
        ) / density_particle / specific_heat_capacity_solid
        dcure = (1 - cure) * scaled_reaction_rate(activation_energy, reaction_rate, temperature_solid)

        return jnp.concatenate(
            jax.tree_util.tree_map(jnp.atleast_1d, (
                dmoisture,
                dtemperature,
                dcure,
                jnp.zeros_like(dtemperature)
                ))
            )


    def solid_model_constant_rate(x, t, p):
        """
        Constant rate dyring equations of solid (single trajectory)
        """
        air, p = p
        moisture_air, temperature_air = air
        moisture_solid, temperature_solid, cure, _ = x # mositure in particle, temperature of solid
        mass_transfer_coefficient = p["mass_transfer_coefficient_constant"] # scale mass transfer coefficient
        reaction_rate = p["reaction_rate"]
        activation_energy = p["activation_energy"]
       
        den_air = density_air(moisture_air, temperature_air)
        y_equil = scaled_moisture_content(temperature_solid, den_air)
        mtc = scaled_mass_transfer_coefficient((moisture_solid, temperature_solid, moisture_air, temperature_air, y_equil), mass_transfer_coefficient, constant)
        dmoisture = - mtc * (y_equil - moisture_air) * den_air / enthalpy_vaporization_water / density_particle * area_volume_ratio
        dcure = (1 - cure) * scaled_reaction_rate(activation_energy, reaction_rate, temperature_solid)

        return jnp.concatenate(
            jax.tree_util.tree_map(jnp.atleast_1d, (
                dmoisture,
                jnp.zeros_like(dmoisture),
                dcure,
                jnp.zeros_like(dmoisture)
                ))
            )


    def air_model(y, t, args):
        """
        Vmapped air model over all discretized points (height)
        """

        x, event_times, p = args

        def _air_model(y, h, args):
            _x, t_event, p = args
            return jax.lax.cond(t <= t_event, lambda : air_model_constant_rate(y, h, (_x, p)), lambda : air_model_falling_rate(y, h, (_x, p)))

        def scan_fun(carry, xs):
            next_carry = odeint_rk_grid(_air_model, carry, jnp.array([0, diff_elem_y]), (*xs, p), dt = 0.01, mxiter = 4)
            return next_carry, next_carry

        _, y_traj = jax.lax.scan(scan_fun, y, xs = (x, event_times))
        return y_traj


    def solid_model(x, t, args):
        """
        Vmapped solid model over all discretized points (height)
        """
        event_times, (yinit, p) = args
        y = air_model(yinit, t, (x, event_times, p))
        dx = jax.vmap(lambda _x, _y, t_event : jax.lax.cond(
            t <= t_event,
            lambda : solid_model_constant_rate(_x, t, (_y, p)),
            lambda : solid_model_falling_rate(_x, t, (_y, p))
        ))(x, y, event_times)
        return dx
   
    def transfer(trans_func, x, t, args) : return trans_func(solid_model, event, x, t, args)

    def _recirculation(moisture_init_guess, p):
        # Root finding function for recirculation
        solid_xinit, yinit, ts, p = p
        yinit_guess = jnp.concatenate((moisture_init_guess, yinit[1:]))
        solution_solid, event_times = odeint_event(solid_model, jax.vmap(event, in_axes = (0, None)), transfer, solid_xinit, ts, (yinit_guess, p), **kwargs)
        solution_air = jax.vmap(air_model, in_axes = (None, 0, (0, None, None)))(yinit_guess, ts, (solution_solid, event_times, p))
        moisture_air = solution_air[1:, :, 0]
        out_moisture = jnp.mean(moisture_air[-1])

        return moisture_init_guess - ((1 - p["recirculation_ratio"]) * p["init_moisture_air"] + p["recirculation_ratio"] * out_moisture)


    xinit = jnp.column_stack(( xinit, parameters["critical_moisture"] * jnp.ones_like(xinit[:, 0]) ))
    init_temp_solid = jax.vmap(solid_temperature_jump, in_axes = (0, None))(xinit, parameters)
    solid_xinit = jnp.column_stack((xinit[:, 0], init_temp_solid, xinit[:, 2:]))

    # ts starts from 0 because odeint starting times requires initial time point. Once the solution is obtained from odeint
    # the initial conditions are discarded.
    ts = jnp.linspace(0, parameters["residence_time"], constant.ntimes).flatten()
    yinit = jnp.concatenate([parameters["init_moisture_air"], parameters["init_temperature_air"]])
       
    yinit_opt = jax.lax.cond(
        recirculation,
        lambda : jnp.concatenate((root_finding_fwd(_recirculation, yinit[:1], (solid_xinit, yinit, ts, parameters), reuse_inverse = reuse_inverse), yinit[1:])),
        lambda : yinit
    )
   
    solution_solid, event_times = odeint_event(solid_model, jax.vmap(event, in_axes = (0, None)), transfer, solid_xinit, ts, (yinit_opt, parameters), **kwargs)
    solution_air = jax.vmap(air_model, in_axes = (None, 0, (0, None, None)))(yinit_opt, ts, (solution_solid, event_times, parameters))

    # remove first element and return (shape = time_span, ny)
    moisture_solid, temperature_solid, cure = solution_solid[1:, :, 0], solution_solid[1:, :, 1], solution_solid[1:, :, 2]
    moisture_air, temperature_air = solution_air[1:, :, 0], solution_air[1:, :, 1]

    return moisture_solid, temperature_solid, cure, moisture_air, temperature_air, event_times
