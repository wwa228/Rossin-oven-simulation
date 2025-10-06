import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import jax.numpy as jnp
import pandas as pd



from test_simulation import (
    constants, Controls, controls, objective, parameters, temperature_init, recirculation, key,
    solid_moisture_mean, solid_moisture_sigma_optimal,
    unscale_states, scale_states,
    temperature_max, temperature_min, moisture_max, moisture_min
)

from oven.oven_dynamics import oven_dynamics
from oven.phy_props import _mass_transfer_coefficient, _heat_transfer_coefficient, moisture_content
from oven.scaling import choose_scaling
from oven.data_structures import Controls, SimulationData, Constants, OdeKwargs, TrainingResults
from oven.plots import plot
from oven.utils import get_coefficients


class SimulationTab:
    def __init__(self, parent):
        self.parent = parent
        self.moisture_cb = None
        self.temp_cb = None
        self.cure_cb = None
        self.air_m_cb = None
        self.air_t_cb = None
        
        self.setup_tab()
        self.initialize_data()
        self.setup_initial_simulation()
    
    def setup_tab(self):
        """Setup the simulation tab with scrollable canvas."""
        self.frame = ttk.Frame(self.parent)
        
        # Create scrollable canvas
        self.main_canvas = tk.Canvas(self.frame, bg="#2b2b2b")
        self.v_scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.main_canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.frame, orient="horizontal", command=self.main_canvas.xview)
        
        self.scrollable_frame = tk.Frame(self.main_canvas, bg="#2b2b2b")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )
        
        # Pack scrollable components
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        
        # Bind mouse wheel events
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind_all("<Shift-MouseWheel>", self._on_shiftwheel)

        # Create GUI components
        self.load_background_image()
        self.zone_entries = self.create_zone_frames()
        self.create_product_properties_frame()
        self.create_buttons()
        self.create_plot_frame()
    
    def initialize_data(self):
        """Initialize simulation parameters and scaling functions."""
        # Air values
        self.init_temperature = jnp.array([490.21, 495.07, 521.69, 554.96, 548.06])  # F
        self.init_temperature_K = (self.init_temperature - 32) * 5 / 9 + 273.15  # K
        self.init_moisture = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])
        self.init_velocity = jnp.array([485, 485, 485, 540.11, 568.61])  # RPM
        self.init_line_speed = [181.11] * 5  # ft/min
        self.converted_velocity = self.convert_units(self.init_velocity) * 5.08e-3 * 2  # m/s
        
        # Setup scaling functions
        (moisture_max, moisture_min, temperature_max, temperature_min), \
        (scale_states, unscale_states, _, _) = choose_scaling("max")
        
        self.moisture_max = moisture_max
        self.moisture_min = moisture_min
        self.temperature_max = temperature_max
        self.temperature_min = temperature_min
        self.scale_states = scale_states
        self.unscale_states = unscale_states
    
    def convert_units(self, y):
        """Convert velocity units."""
        x = (((y - 650) / (485 - 650)) * 22000)
        w = ((1 - (y - 650) / (485 - 650)) * 32000)
        result = (x + w) / (24 * 69)
        return result
    
    def load_background_image(self):
        """Load and display background image."""
        try:
            image_path = "C:/Users/wwa228/Downloads/Oven.png"
            img = Image.open(image_path)
            img = img.resize((800, 150))
            bg_img = ImageTk.PhotoImage(img)
            
            image_frame = tk.Frame(self.scrollable_frame, bg="#D80909")
            image_frame.pack(side="top", fill="x")
            background = tk.Label(image_frame, image=bg_img)
            background.image = bg_img
            background.pack()
            
        except Exception as e:
            print("Couldn't load background image:", e)
    
    def create_zone_frames(self):
        """Create input frames for each zone."""
        zone_frame = tk.Frame(self.scrollable_frame, bg="#8f2727")
        zone_frame.pack(pady=10)
        
        zone_entries = []
        for i in range(5):
            z = tk.LabelFrame(zone_frame, text=f"Zone{i+1}", labelanchor="n", padx=10, pady=10)
            z.pack(side="left", padx=10)
            
            entries = {}
            for j, param in enumerate(['Temperature', 'Moisture', 'line_speed', 'velocity']):
                tk.Label(z, text=f"{param}: ").grid(row=j, column=0)
                entry = tk.Entry(z, width=5)
                entry.grid(row=j, column=1)
                
                if param == 'Temperature':
                    unit_var = tk.StringVar(value='K')
                    unit_menu = ttk.Combobox(z, textvariable=unit_var,
                                           values=['°C', '°F', 'K'],
                                           width=5, state='readonly')
                    unit_menu.grid(row=j, column=2)
                    entries[param] = (entry, unit_var)
                else:
                    unit_var = tk.StringVar()
                    single_unit = {
                        'Moisture': 'kg/kg',
                        'line_speed': 'ft/min',
                        'velocity': 'RPM'
                    }.get(param, '')
                    unit_var.set(single_unit)
                    unit_menu = ttk.Combobox(z, textvariable=unit_var,
                                           values=[single_unit],
                                           width=6, state='readonly')
                    unit_menu.grid(row=j, column=2)
                    entries[param] = (entry, unit_var)
            zone_entries.append(entries)
        return zone_entries
    
    def create_product_properties_frame(self):
        """Create frame displaying product properties."""
        props_frame = tk.LabelFrame(
            self.scrollable_frame, text="Product Properties",
            bg="#d0c3c3", fg="white",
            font=("Arial", 12, "bold"),
            labelanchor="n", padx=10, pady=10
        )
        props_frame.pack(padx=10, pady=10, fill="x")
        
        desired_keys = {
            "density_particle": "Density Particle",
            "density_product": "Density Product",
            "enthalpy_vaporization_water": "Enthalpy of Vaporization of Water",
            "nzones": "Number of Zones",
            "reverse_zone": "Reverse Zones",
            "product_height": "Product Height",
            "radius_particle": "Particle Radius",
            "specific_heat_capacity_air": "Specific Heat Capacity of Air",
            "specific_heat_capacity_solid": "Specific Heat Capacity of Solid",
            "voidage": "Voidage"
        }
        
        row_index = 0
        for attr, display_name in desired_keys.items():
            if hasattr(constants, attr):
                value = getattr(constants, attr)
                label = tk.Label(props_frame, text=display_name, width=30, anchor="w",
                               bg="#2b2b2b", fg="white", font=("Arial", 10))
                label.grid(row=row_index, column=0, sticky="w", padx=5, pady=2)
                
                entry_var = tk.StringVar(value=str(value))
                entry = tk.Entry(props_frame, textvariable=entry_var, state="readonly",
                               readonlybackground="#3e3e3e", fg="white", relief="flat")
                entry.grid(row=row_index, column=1, sticky="ew", padx=5, pady=2)
                row_index += 1
    
    def create_buttons(self):
        """Create update plots and export buttons side by side."""
        # Container frame for buttons
        button_container = tk.Frame(self.scrollable_frame, bg="#2b2b2b")
        button_container.pack(padx=10, pady=10, fill="x")
        
        # Update button
        self.update_button = tk.Button(
            button_container, text="Update Plots",
            command=self.update_plots_from_entries,
            bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
            padx=20, pady=10, relief="raised", bd=3, cursor="hand2"
        )
        self.update_button.pack(side="left", padx=(0, 5), expand=True, fill="x")
        
        # Export button
        self.export_button = tk.Button(
            button_container,
            text="Export Data",
            bg="#2196F3", fg="white",
            font=("Arial", 12, "bold"),
            command=self.export_data,
            padx=20, pady=10, relief="raised", bd=3, cursor="hand2"
        )
        self.export_button.pack(side="left", padx=(5, 0), expand=True, fill="x")
        
        # Add hover effects
        def on_enter_update(e):
            self.update_button.config(bg='#45a049')
        def on_leave_update(e):
            self.update_button.config(bg='#4CAF50')
        def on_enter_export(e):
            self.export_button.config(bg='#1976D2')
        def on_leave_export(e):
            self.export_button.config(bg='#2196F3')
        
        self.update_button.bind("<Enter>", on_enter_update)
        self.update_button.bind("<Leave>", on_leave_update)
        self.export_button.bind("<Enter>", on_enter_export)
        self.export_button.bind("<Leave>", on_leave_export)
        
    def create_plot_frame(self):
        """Create plotting frame and canvas."""
        graph_frame = tk.Frame(self.scrollable_frame, bg="#273c8f")
        graph_frame.pack(fill="both", expand=True)
        
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))
        for ax in self.axs.flat:
            ax.set_facecolor("#f5f5f5")
        
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.3, hspace=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    
    def setup_initial_simulation(self):
        """Setup initial state with empty plots - don't run simulation yet."""
        # Store the initial values for reference, but don't run simulation
        self.init_temperature = jnp.array([490.21, 495.07, 521.69, 554.96, 548.06])  # F
        self.init_temperature_K = (self.init_temperature - 32) * 5 / 9 + 273.15  # K
        self.init_moisture = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])
        self.init_velocity = jnp.array([485, 485, 485, 540.11, 568.61])  # RPM
        self.init_line_speed = [181.11] * 5  # ft/min
        self.converted_velocity = self.convert_units(self.init_velocity) * 5.08e-3 * 2  # m/s

        #initialize empty data arrays
        self.moisture_solid = None
        self.temperature_solid = None
        self.cure = None
        self.moisture_air = None
        self.temperature_air = None
        self.time_span = None

        # Populate zone entries with initial values but don't run simulation

        self.update_zone_entries(self.init_temperature, self.init_moisture, self.init_line_speed, self.init_velocity)

        #Show empty plots
        self.plot_empty_data()
        self.canvas.draw()
        '''
        init_controls = Controls(inputs={
            "init_temperature_air": self.scale_states(self.init_temperature_K, self.temperature_max, self.temperature_min),
            "init_velocity_air": self.converted_velocity,
            "init_moisture_air": self.scale_states(self.init_moisture, self.moisture_max, self.moisture_min),
            "residence_time": jnp.atleast_1d(jnp.round(24 * 60. / self.init_line_speed[0])),
            "recirculation_ratio": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        })
        
        _, _, moisture_solid_scaled, temperature_solid_scaled, cure, moisture_air, temperature_air, ts_event = objective(
            parameters, solid_moisture_mean, solid_moisture_sigma_optimal,
            constants, init_controls, temperature_init, recirculation, key
        )
        
        self.moisture_solid = np.asarray(self.unscale_states(moisture_solid_scaled, self.moisture_max, self.moisture_min))
        self.temperature_solid = np.asarray(self.unscale_states(temperature_solid_scaled, self.temperature_max, self.temperature_min))
        self.cure = cure
        self.moisture_air = moisture_air
        self.temperature_air = temperature_air
        self.time_span = jnp.arange(init_controls["residence_time"][0] * constants.nzones).flatten()
        
        self.plot_data()
        self.update_zone_entries(self.init_temperature, self.init_moisture, self.init_line_speed, self.init_velocity)
        self.canvas.draw()
        '''
    def plot_empty_data(self):
        """Display empty plots with instructions."""
        # Remove existing colorbars
        for cb in [self.moisture_cb, self.temp_cb, self.cure_cb, self.air_m_cb, self.air_t_cb]:
            if cb:
                cb.remove()
        
        # Subplot references
        ax_moisture = self.axs[0, 0]
        ax_temp = self.axs[0, 1]
        ax_cure = self.axs[0, 2]
        ax_air_moisture = self.axs[1, 0]
        ax_air_temp = self.axs[1, 1]
        
        # Clear all plots
        for ax in [ax_moisture, ax_temp, ax_cure, ax_air_moisture, ax_air_temp]:
            ax.clear()
        
        # Display "no data" messages
        ax_moisture.text(0.5, 0.5, "Enter zone values\nand click 'Update Plots'", 
                        ha='center', va='center', transform=ax_moisture.transAxes, fontsize=12)
        ax_moisture.set_title("Solid Moisture Content")
        ax_moisture.set(xlabel="Time", ylabel="Product Height")
        
        ax_temp.text(0.5, 0.5, "Enter zone values\nand click 'Update Plots'", 
                    ha='center', va='center', transform=ax_temp.transAxes, fontsize=12)
        ax_temp.set_title("Solid Temperature")
        ax_temp.set(xlabel="Time", ylabel="Product Height")
        
        ax_cure.text(0.5, 0.5, "Enter zone values\nand click 'Update Plots'", 
                    ha='center', va='center', transform=ax_cure.transAxes, fontsize=12)
        ax_cure.set_title("Degree of Cure")
        ax_cure.set(xlabel="Time", ylabel="Product Height")
        
        ax_air_moisture.text(0.5, 0.5, "Enter zone values\nand click 'Update Plots'", 
                            ha='center', va='center', transform=ax_air_moisture.transAxes, fontsize=12)
        ax_air_moisture.set_title("Air Moisture")
        ax_air_moisture.set(xlabel="Time", ylabel="Product Height")
        
        ax_air_temp.text(0.5, 0.5, "Enter zone values\nand click 'Update Plots'", 
                        ha='center', va='center', transform=ax_air_temp.transAxes, fontsize=12)
        ax_air_temp.set_title("Air Temperature")
        ax_air_temp.set(xlabel="Time", ylabel="Product Height")
        
        # Hide unused 6th subplot
        self.axs[1, 2].axis("off")
    
    def plot_data(self):
        """Plot simulation data on the canvas."""
        # Remove existing colorbars
        for cb in [self.moisture_cb, self.temp_cb, self.cure_cb, self.air_m_cb, self.air_t_cb]:
            if cb:
                cb.remove()
        
        time_span = jnp.arange(controls["residence_time"][0] * constants.nzones).flatten()
        height_span = jnp.linspace(0, constants.product_height, constants.ny)
        x, y = jnp.meshgrid(time_span, height_span)
        
        # Subplot references
        ax_moisture = self.axs[0, 0]
        ax_temp = self.axs[0, 1]
        ax_cure = self.axs[0, 2]
        ax_air_moisture = self.axs[1, 0]
        ax_air_temp = self.axs[1, 1]
        
        # Clear previous plots
        for ax in [ax_moisture, ax_temp, ax_cure, ax_air_moisture, ax_air_temp]:
            ax.clear()
        
        # Solid moisture
        moisture_plot = ax_moisture.pcolormesh(x, y, self.moisture_solid[:-1, :-1].T, cmap="Greens_r", shading="auto")
        self.moisture_cb = self.fig.colorbar(moisture_plot, ax=ax_moisture)
        self.moisture_cb.set_label("Moisture (kg/kg)")
        ax_moisture.set(xlabel="Time", ylabel="Product height", title="Solid Moisture content")
        ax_moisture.set_ylim(0, 0.1)
        ax_moisture.set_xlim(0, self.time_span[-1])
        
        # Solid temperature
        temp_plot = ax_temp.pcolormesh(x, y, self.temperature_solid[:-1, :-1].T, cmap="coolwarm", shading="auto")
        self.temp_cb = self.fig.colorbar(temp_plot, ax=ax_temp)
        self.temp_cb.set_label("Temperature (K)")
        ax_temp.set(xlabel="Time", ylabel="Product height", title="Solid Temperature")
        ax_temp.set_ylim(0, 0.1)
        ax_temp.set_xlim(0, self.time_span[-1])
        
        # Cure Plot
        cure_plot = ax_cure.pcolormesh(x, y, self.cure[:-1, :-1].T, cmap="viridis", shading="auto")
        self.cure_cb = self.fig.colorbar(cure_plot, ax=ax_cure)
        ax_cure.set(xlabel="Time", ylabel="Product height", title="Degree of cure")
        self.cure_cb.set_label("Cure (%)")
        
        # Air Moisture
        air_m_plot = ax_air_moisture.pcolormesh(x, y, self.moisture_air[:-1, :-1].T, cmap="Greens_r", shading="auto")
        self.air_m_cb = self.fig.colorbar(air_m_plot, ax=ax_air_moisture)
        ax_air_moisture.set(xlabel="Time", ylabel="Product height", title="Air Moisture")
        self.air_m_cb.set_label("Air Moisture (kg/kg)")
        
        # Air Temperature
        air_t_plot = ax_air_temp.pcolormesh(x, y, self.temperature_air[:-1, :-1].T, cmap="coolwarm", shading="auto")
        self.air_t_cb = self.fig.colorbar(air_t_plot, ax=ax_air_temp)
        ax_air_temp.set(xlabel="Time", ylabel="Product height", title="Air Temperature")
        self.air_t_cb.set_label("Air Temperature (K)")
        
        # Add zone separation lines
        for j in range(constants.nzones - 1):
            vline_x = (j + 1) * controls["residence_time"][0]
            for ax in [ax_moisture, ax_temp, ax_cure, ax_air_moisture, ax_air_temp]:
                ax.axvline(vline_x, color="k", linestyle="--", linewidth=0.8)
        
        # Hide unused 6th subplot
        self.axs[1, 2].axis("off")
    
    def update_zone_entries(self, T_data, M_data, L_data, V_data):
        """Update zone entry fields with new data."""
        for i in range(5):
            try:
                T_entry, T_unit = self.zone_entries[i]['Temperature']
                M_entry, _ = self.zone_entries[i]['Moisture']
                L_entry, _ = self.zone_entries[i]['line_speed']
                V_entry, _ = self.zone_entries[i]['velocity']
                
                T_entry.delete(0, tk.END)
                M_entry.delete(0, tk.END)
                L_entry.delete(0, tk.END)
                V_entry.delete(0, tk.END)
                
                unit = T_unit.get()
                T_val = T_data[i]
                if unit == '°F':
                    display_temp = (T_val - 273.15) * 9/5 + 32
                elif unit == '°C':
                    display_temp = T_val - 273.15
                else:
                    display_temp = T_val
                
                T_entry.insert(0, str(round(display_temp, 2)))
                T_unit.set('K')
                M_entry.insert(0, str(round(M_data[i], 5)))
                L_entry.insert(0, str(round(L_data[i], 2)))
                V_entry.insert(0, str(round(V_data[i], 2)))
                
            except Exception as e:
                print(f"[ERROR updating zone {i} entries] -> {e}")
    
    def update_plots_from_entries(self):
        """Update plots based on current entry values."""
        try:
            updated_T = []
            updated_M = []
            updated_L = []
            updated_V = []
            
            for i, zone in enumerate(self.zone_entries):
                T_entry, T_unit = zone['Temperature']
                temp_str = T_entry.get()
                unit = T_unit.get()
                T_val_input = float(temp_str)
                
                if unit == '°F':
                    T_val_input = (T_val_input - 32) * 5/9 + 273.15
                elif unit == '°C':
                    T_val_input = T_val_input + 273.15
                
                updated_T.append(T_val_input)
                T_entry.delete(0, tk.END)
                T_entry.insert(0, str(round(T_val_input, 2)))
                T_unit.set('K')  # Change dropdown to 'K'
                
                updated_M.append(float(zone['Moisture'][0].get()))
                updated_L.append(float(zone['line_speed'][0].get()))
                updated_V.append(float(zone['velocity'][0].get()))
            
            updated_controls = Controls(inputs={
                "init_temperature_air": self.scale_states(jnp.array(updated_T), self.temperature_max, self.temperature_min),
                "init_moisture_air": self.scale_states(jnp.array(updated_M), self.moisture_max, self.moisture_min),
                "init_velocity_air": self.convert_units(jnp.array(updated_V)) * 5.08e-3 * 2,
                "residence_time": jnp.atleast_1d(jnp.round(24 * 60. / updated_L[0])),
                "recirculation_ratio": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            })
            
        except Exception as e:
            print(f"Error reading entries or building simulation inputs: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            #Run the simulation with the user inputs
            _, _, moisture_solid_scaled, temperature_solid_scaled, cure, moisture_air, temperature_air, ts_event = objective(
                parameters, solid_moisture_mean, solid_moisture_sigma_optimal,
                constants, updated_controls, temperature_init, recirculation, key
            )
            
            self.moisture_solid = np.asarray(self.unscale_states(moisture_solid_scaled, self.moisture_max, self.moisture_min))
            self.temperature_solid = np.asarray(self.unscale_states(temperature_solid_scaled, self.temperature_max, self.temperature_min))
            self.cure = cure
            self.moisture_air = moisture_air
            self.temperature_air = temperature_air
            self.time_span = jnp.arange(updated_controls["residence_time"][0] * constants.nzones).flatten()
            
            #plot actual data
            self.plot_data()
            self.update_zone_entries(updated_T, updated_M, updated_L, updated_V)
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error re-running simulation or updating plot: {e}")
            import traceback
            traceback.print_exc()
    
    def export_data(self):
        """Export simulation data to file."""
        #Check if simulation has been run

        if self.moisture_solid is None:
            messagebox.showwarning("No Data", "Please run the simulation first by clicking on update plots")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:  # User cancelled
            return
        
        try:
            data = {
                "Time": self.time_span,
                "Moisture": self.moisture_solid.flatten(),
                "Temperature": self.temperature_solid.flatten(),
                "Cure": self.cure.flatten(),
                "Air Moisture": self.moisture_air.flatten(),
                "Air Temperature": self.temperature_air.flatten()
            }
            
            df = pd.DataFrame(data)
            
            if file_path.endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
            
            print(f"Data exported successfully to {file_path}")
            
        except Exception as e:
            print("Error exporting data:", e)
    
    def _on_mousewheel(self, event):
        """Handle vertical mouse wheel scrolling."""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_shiftwheel(self, event):
        """Handle horizontal mouse wheel scrolling."""
        self.main_canvas.xview_scroll(int(-1*(event.delta/120)), "units")