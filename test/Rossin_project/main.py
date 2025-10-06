import tkinter as tk
from tkinter import ttk
from .simulation import SimulationTab
from .training import TrainingTab
from .optimization import OptimizationTab
import time


class OvenControlPanel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Oven Control Panel")
        self.root.geometry("900x600")
        self.root.configure(bg="#2b2b2b")
        
        start_time = time.time()
        print(" Initializing Oven Control Panel...")

        self.setup_notebook()
        self.setup_tabs()

        elapsed_time = time.time() - start_time
        print(f"✅ GUI ready in {elapsed_time:.1f} seconds!")

    def setup_notebook(self):
        """Initialize the main notebook widget for tabs."""
        self.notebook = ttk.Notebook(self.root)
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding =[0,10,0,10] )
        self.notebook.pack(fill='both', expand=True)
    
    def setup_tabs(self):
        """Create and configure all tabs."""
        tab_start = time.time()
        print("Loading Simulation Tab...")

        # Simulation Tab
        self.simulation_tab = SimulationTab(self.notebook)
        self.notebook.add(self.simulation_tab.frame, text=" 🧪Simulation ")

        # Training Tab
        print("Loading Training Tab...")
        self.training_tab = TrainingTab(self.notebook)
        self.notebook.add(self.training_tab.frame, text=" Training ")

        print(" Loading Optimization Tab...")
        # Energy Optimization Tab
        self.optimization_tab = OptimizationTab(self.notebook)
        self.notebook.add(self.optimization_tab.frame, text=" Energy Optimization ")

        tab_elapsed = time.time() - tab_start
        print(f" All tabs loaded in {tab_elapsed:.1f}s")

    def run(self):
        """Start the main GUI event loop."""
        self.root.mainloop()


def main():
    """Main entry point for the application."""
    print(" Initializing Oven Control Panel...")
    app = OvenControlPanel()
    app.run()


if __name__ == "__main__":
    main()