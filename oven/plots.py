import os
import operator
from typing import Optional, Tuple, Dict

import jax.numpy as jnp
from jax import tree_util
import matplotlib.pyplot  as plt
import scienceplots


from .utils import check_dir


def plot(
        data, result, dimension_span : Tuple, temperature_target : Optional[dict] = None, 
        plot_names : Optional[list] = None, dir : str = "."
    ) -> None:
    
    # moisture_and_temperature dimesions = (ntraj, time_span, height_span)
    moisture_solid, temperature_solid, moisture_air, temperature_air = data.moisture_solid, data.temperature_solid, data.moisture_air, data.temperature_air
    event_times, htc, mtc = data.event_times, data.heat_transfer_coefficient, data.mass_transfer_coefficient
    training_loss, testing_loss, learning_rate = result.training_loss, result.testing_loss, result.learning_rate
    time_span, height_span, nzones, residence_time = dimension_span
    solid_plot, mesh_plot_solid, mesh_plot_air, mesh_plot_coeff = (
        plot_names if plot_names or len(plot_names) == 4 else  
        ["event_solid", "event_solid_mesh", "event_air_mesh", "event_coeff_mesh"])

    ny = len(height_span)

    # if path does not exists create new dir
    check_dir(dir)
    
    # plotting 2d temperature values
    keys = []
    with plt.style.context(["science", "notebook", "bright"]):
        fig, ax = plt.subplots(2, 2, figsize = (20, 15), gridspec_kw = {"hspace" : 0.3})
                
        # mid predictions
        _line = ax[0, 1].plot(time_span, temperature_solid[0, :, ny//2], label = "Mean")
        ax[0, 1].fill_between(time_span, temperature_solid[-1, :, ny//2], temperature_solid[1, :, ny//2], alpha = 0.4, color = _line[0].get_color())

        # bottom predictions
        ax[1, 0].plot(time_span, temperature_solid[0, :, 1], label = "Mean")
        ax[1, 0].fill_between(time_span, temperature_solid[-1, :, 1], temperature_solid[1, :, 1], alpha = 0.4, color = _line[0].get_color())

        # top predictions
        ax[1, 1].plot(time_span, temperature_solid[0, :, -2], label = "Mean")
        ax[1, 1].fill_between(time_span, temperature_solid[-1, :, -2], temperature_solid[1, :, -2], alpha = 0.4, color = _line[0].get_color())    

        # other key predictions
        if temperature_target :

            keys = list(temperature_target.keys())
            for key in keys :
                ax[0, 0].plot(time_span, temperature_solid[0, :, key], label = "Mean")
                ax[0, 0].fill_between(time_span, temperature_solid[-1, :, key], temperature_solid[1, :, key], alpha = 0.4, color = _line[0].get_color())

            for key, value in temperature_target.items():
                ax[0, 0].plot(time_span, value, "o", label = f"Target")
        
        # plot vertical lines for each zones
        for j in range(nzones - 1):
            ax[0, 0].axvline((j + 1)*residence_time, color = "k")
            ax[0, 1].axvline((j + 1)*residence_time, color = "k")
            ax[1, 0].axvline((j + 1)*residence_time, color = "k")
            ax[1, 1].axvline((j + 1)*residence_time, color = "k")

        for key in keys : 
            ax[0, 0].set(xlabel = "Time (sec)", ylabel = "Temperature (K)")
        ax[0, 1].set(xlabel = "Time (sec)", ylabel = "Temperature (K)", title = "Predicted (mid)")
        ax[1, 0].set(xlabel = "Time (sec)", ylabel = "Temperature (K)", title = "Predicted (bottom)")
        ax[1, 1].set(xlabel = "Time (sec)", ylabel = "Temperature (K)", title = "Predicted (top)")

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()
        
        plt.savefig(os.path.join(dir, solid_plot) + "Temperature.png")
        plt.close()

    # plotting 2d moisture values
    with plt.style.context(["science", "notebook", "bright"]):
        fig, ax = plt.subplots(2, 2, figsize = (20, 15), gridspec_kw = {"hspace" : 0.3})
                
        # mid predictions
        _line = ax[0, 1].plot(time_span, moisture_solid[0, :, ny//2], label = "Mean")
        ax[0, 1].fill_between(time_span, moisture_solid[-1, :, ny//2], moisture_solid[1, :, ny//2], alpha = 0.4, color = _line[0].get_color())

        # bottom predictions
        ax[1, 0].plot(time_span, moisture_solid[0, :, 1], label = "Mean")
        ax[1, 0].fill_between(time_span, moisture_solid[-1, :, 1], moisture_solid[1, :, 1], alpha = 0.4, color = _line[0].get_color())

        # top predictions
        ax[1, 1].plot(time_span, moisture_solid[0, :, -2], label = "Mean")
        ax[1, 1].fill_between(time_span, moisture_solid[-1, :, -2], moisture_solid[1, :, -2], alpha = 0.4, color = _line[0].get_color())   

        # other key predictions
        if temperature_target :

            for key in keys :
                ax[0, 0].plot(time_span, moisture_solid[0, :, key], label = "Mean")
                ax[0, 0].fill_between(time_span, moisture_solid[-1, :, key], moisture_solid[1, :, key], alpha = 0.4, color = _line[0].get_color())

        # plot vertical lines for each zones
        for j in range(nzones - 1):
            ax[0, 0].axvline((j + 1)*residence_time, color = "k")
            ax[0, 1].axvline((j + 1)*residence_time, color = "k")
            ax[1, 0].axvline((j + 1)*residence_time, color = "k")
            ax[1, 1].axvline((j + 1)*residence_time, color = "k")

        for key in keys : 
            ax[0, 0].set(xlabel = "Time (sec)", ylabel = "Moisture (Kg/Kg)")
        ax[0, 1].set(xlabel = "Time (sec)", ylabel = "Moisture (Kg/Kg)", title = "Predicted (mid)")
        ax[1, 0].set(xlabel = "Time (sec)", ylabel = "Moisture (Kg/Kg)", title = "Predicted (bottom)")
        ax[1, 1].set(xlabel = "Time (sec)", ylabel = "Moisture (Kg/Kg)", title = "Predicted (top)")

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()
        
        plt.savefig(os.path.join(dir, solid_plot) + "Moisture.png")
        plt.close()
 
    # plotting results 3d (solid)
    x, y = jnp.meshgrid(time_span, height_span)
    with plt.style.context(["science", "notebook", "bright"]):
        fig, ax = plt.subplots(1, 2, figsize = (15, 7))
        c = ax[0].pcolor(x, y, moisture_solid[0][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
        fig.colorbar(c, ax = ax[0])
        ax[0].set(xlabel = "Time", ylabel = "Product height", title = "Moisture")

        c = ax[1].pcolor(x, y, temperature_solid[0][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
        fig.colorbar(c, ax = ax[1])
        ax[1].set(xlabel = "Time", ylabel = "Product height", title = "Temperature")

        # plot event times
        ax[0].step(event_times, height_span, label = "Event Times", color = "y", linewidth = 3.)
        ax[1].step(event_times, height_span, label = "Event Times", color = "y", linewidth = 3.)

        # plot vertical lines for each zone
        for j in range(nzones - 1):
            ax[0].axvline((j + 1)*residence_time, color = "k")
            ax[1].axvline((j + 1)*residence_time, color = "k")

        plt.savefig(os.path.join(dir, mesh_plot_solid) + ".png")
        plt.close()

    # plotting results 3d (air)
    with plt.style.context(["science", "notebook", "bright"]):
        fig, ax = plt.subplots(1, 2, figsize = (15, 7))
        c = ax[0].pcolor(x, y, moisture_air[0][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
        fig.colorbar(c, ax = ax[0])
        ax[0].set(xlabel = "Time", ylabel = "Product height", title = "Moisture content")

        c = ax[1].pcolor(x, y, temperature_air[0][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
        fig.colorbar(c, ax = ax[1])
        ax[1].set(xlabel = "Time", ylabel = "Product height", title = "Temperature")

        # plot event times
        ax[0].step(event_times, height_span, label = "Event Times", color = "y", linewidth = 3.)
        ax[1].step(event_times, height_span, label = "Event Times", color = "y", linewidth = 3.)

        # plot vertical lines for each zone
        for j in range(nzones - 1):
            ax[0].axvline((j + 1)*residence_time, color = "k")
            ax[1].axvline((j + 1)*residence_time, color = "k")

        plt.savefig(os.path.join(dir, mesh_plot_air) + ".png")
        plt.close()

    # plotting results 3d (heat and mass transfer coefficient)
    if htc is not None and mtc is not None:
        with plt.style.context(["science", "notebook", "bright"]):
            fig, ax = plt.subplots(1, 2, figsize = (15, 7))
            c = ax[0].pcolor(x, y, mtc[0][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
            fig.colorbar(c, ax = ax[0])
            ax[0].set(xlabel = "Time", ylabel = "Product height", title = "Mass Transfer Coefficient")

            c = ax[1].pcolor(x, y, htc[0][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
            fig.colorbar(c, ax = ax[1])
            ax[1].set(xlabel = "Time", ylabel = "Product height", title = "Heat Transfer Coefficient")

            # plot event times
            ax[0].step(event_times, height_span, label = "Event Times", color = "y", linewidth = 3.)
            ax[1].step(event_times, height_span, label = "Event Times", color = "y", linewidth = 3.)

            # plot vertical lines for each zone
            for j in range(nzones - 1):
                ax[0].axvline((j + 1)*residence_time, color = "k")
                ax[1].axvline((j + 1)*residence_time, color = "k")

            plt.savefig(os.path.join(dir, mesh_plot_coeff) + ".png")
            plt.close()

    # plotting training and testing losses 
    if training_loss is not None and testing_loss is not None and learning_rate is not None :
        with plt.style.context(["science", "notebook", "bright"]) :
            fig, ax = plt.subplots(1, 2, figsize = (20, 10))
            if training_loss is not None and testing_loss is not None :
                ax[0].plot(training_loss, "o", label = "Training")
                ax[0].plot(testing_loss, "o", label = "Testing")
                ax[0].set(xlabel = "Epoch", ylabel = "Loss", yscale = "symlog") # loss can be negative
                ax[0].legend(frameon = True)

            if learning_rate is not None :
                ax[1].plot(learning_rate)
                ax[1].set(xlabel = "Epoch", ylabel = "LearningRate")    

            
            plt.savefig(os.path.join(dir, "TrainingResults") + ".png")
            plt.close()


def plot_recirculation(energy : jnp.ndarray, cure : jnp.ndarray, ratio : jnp.ndarray, dir : str = "."):

    with plt.style.context(["science", "notebook", "bright"]):

        fig, ax = plt.subplots(1, 1, figsize = (15, 7))
        ax2 = ax.twinx()

        color1, color2 = plt.cm.viridis([0.5, 0.8])
        p1 = ax.plot(ratio, energy, "o-", color = color1, label = "Energy Consumption")
        p2 = ax2.plot(ratio, cure / jnp.max(cure), "o-", color = color2, label = "Performance Index") # scale between (1, 0)

        ax.set(xlabel = "Recirculation Ratio", ylabel = "Energy Consumption", yscale = "log")
        ax2.set(xlabel = "Recirculation Ratio", ylabel = "Performance Index", ylim = (0, 1.1))

        ax.legend(handles = p1 + p2)
        
        plt.savefig(os.path.join(dir, "recir") + ".png")
        plt.close()