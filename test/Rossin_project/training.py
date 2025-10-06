import tkinter as tk
from tkinter import ttk, messagebox
import subprocess, threading
import os
import time
import queue
import glob


class TrainingTab:
    """
    Training tab for the oven control panel.
    Collects training parameters and streams training logs.
    """

    #defining a constructor for the class
    #runs automatically when an object of the class is created
    def __init__(self, parent):
        self.parent = parent # container widget(tk window)
        self.process = None # tracking the  background process(training)
        self.log_queue = queue.Queue() # helps to pass messages between background process and GUI. To avoid freezing
        self.setup_styles()
        self.setup_tab() # calls method to build GUI elements(buttons)

    def setup_tab(self):
        """Setup the training tab"""
        # Main frame with dark theme
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill='both', expand=True)
        self.frame.configure(style='Dark.TFrame')

        # Create canvas and scrollbars
        self.main_canvas = tk.Canvas(self.frame, bg="#2b2b2b", highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.main_canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.frame, orient="horizontal", command=self.main_canvas.xview)
        
        # Configure canvas scrolling
        self.main_canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )
        
        # Pack scrollbars FIRST, then canvas
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        # Create scrollable frame inside canvas
        self.scrollable_frame = tk.Frame(self.main_canvas, bg="#2b2b2b")
        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bind configuration
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        # Bind canvas resize to update window width
        self.main_canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bind mouse wheel events
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind("<Shift-MouseWheel>", self._on_shiftwheel)

        # Bind all that is within it as well
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind("<Shift-MouseWheel>", self._on_shiftwheel)
        
        self.create_training_widgets()
        
        # Cleanup on close
        self.parent.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.cleanup)

    def _on_canvas_configure(self, event):
        """Update the scroll window to match canvas width"""
        self.main_canvas.itemconfig(self.canvas_window, width=event.width)
    def _on_mousewheel(self, event):
        """Handle vertical mouse wheel scrolling."""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_shiftwheel(self, event):
        """Handle horizontal mouse wheel scrolling."""
        self.main_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
    def setup_styles(self):
        style = ttk.Style()

        # Configure styles for dark theme 
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Header.TLabelframe', background='#8f2727',foreground='white', 
                       borderwidth=2,relief='raised')
        
        style.configure('Header.TLabelframe.Label',background='#8f2727',
                        foreground='white', font=('Arial', 10, 'bold'))
        
        style.configure('Param.TLabel', background='#2b2b2b',foreground='#ffffff',
                         font=('Arial', 9))
        
        style.configure('Param.TEntry',fieldbackground='#404040',foreground='white',
                        borderwidth=1,insertcolor='white')
        
        style.configure('Success.TButton',background='#4CAF50',foreground='white',
                        font=('Arial', 11, 'bold'),padding=(10, 5))
        
        style.configure('Danger.TButton', background='#f44336',foreground='white',
                        font=('Arial', 11, 'bold'),padding=(10, 5))

    def create_training_widgets(self):   
        # Main container with padding
        main_container = tk.Frame(self.scrollable_frame, bg='#2b2b2b')
        main_container.pack(fill='both', expand=True, padx=20, pady=15)
        
        # Title header
        title_frame = tk.Frame(main_container, bg='#D80909', relief='raised', bd=2)
        title_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(title_frame, text="TRAINING CONTROL PANEL",bg='#D80909', 
                              fg='white',font=('Arial', 16, 'bold'),pady=10).pack()

        # Parameters section
        params_frame = tk.LabelFrame(main_container,text="Training Parameters",
                                     bg='#8f2727',fg='white',font=('Arial', 12, 'bold'),
                                   relief='raised',bd=2,labelanchor='n')
        params_frame.pack(fill='x', pady=(0, 15))
        
        
        self.create_parameter_grid(params_frame)
        
        # Log section
        log_frame = tk.LabelFrame(main_container,text="Training Logs",bg='#273c8f',
                                fg='white',font=('Arial', 12, 'bold'),relief='raised',
                                bd=2,labelanchor='n')
        log_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        self.create_log_section(log_frame)
        
        # Control buttons section
        button_frame = tk.Frame(main_container, bg='#2b2b2b')
        button_frame.pack(fill='x')
        
        self.create_control_buttons(button_frame)

    def create_parameter_grid(self, parent):
        '''Creates parameter input grid.'''
        inner_frame = tk.Frame(parent, bg='#8f2727')
        inner_frame.pack(fill='x', padx=15, pady=15)
        
        widgets_info = [
            ("Products (Train):", "entry_train", "R13SSR4"),
            ("Products (Test):", "entry_test", "R13SSR4"),
            ("Iterations:", "entry_iter", "100"),
            ("Atol:", "entry_atol", "1e-8"),
            ("Rtol:", "entry_rtol", "1e-6"),
            ("mxstep:", "entry_mxstep", "10000"),
            ("Message:", "entry_msg", ""),
            ("Load Dir:", "entry_load_dir", "."),
            ("Optimize:", "optimize_var", ["1", "0"]),
            ("Recirculation:", "recirculation_var", ["0", "1"]),
            ("Learning Rate:", "entry_lr", "0.001"),
            ("Generality Check:", "entry_gen", "200")
        ]

        self.entries = {}
        self.vars = {}
        
        # Create grid layout (2 columns)
        for i, (label_text, var_name, default_val) in enumerate(widgets_info):
            # Create 2-column grid layout
            row = i // 2      # Integer division for row index (0,0,1,1,2,2,...)
            col = (i % 2) * 3 # Modulo for column index, multiplied by 3 for spacing (0,3,0,3,...)
                        
            # Parameter label
            label = tk.Label(inner_frame, text=label_text,bg='#8f2727',
                            fg='white',font=('Arial', 9, 'bold'),anchor='w')
            label.grid(row=row, column=col, sticky="w", padx=(5, 10), pady=5)
            
            if isinstance(default_val, list): # default value is a list
                var = tk.StringVar(value=default_val[0])
                self.vars[var_name] = var
                combo = ttk.Combobox(inner_frame, textvariable=var,values=default_val,
                                   width=12,font=('Arial', 9))
                combo.configure(style='Param.TCombobox')
                combo.grid(row=row, column=col+1, sticky="w", padx=(0, 20), pady=5)
            else:  # Entry
                entry = tk.Entry(inner_frame,width=15,bg='#404040',fg='white',font=('Arial', 9),
                                relief='solid',bd=1,insertbackground='white')
                entry.insert(0, default_val)
                self.entries[var_name] = entry
                entry.grid(row=row, column=col+1, sticky="w", padx=(0, 20), pady=5)

    def create_log_section(self, parent):
        """Create a log display section."""
        log_container = tk.Frame(parent, bg='#273c8f')
        log_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Log text with scrollbar
        log_frame = tk.Frame(log_container, bg='#273c8f')
        log_frame.pack(fill='both', expand=True)
        
        # Create text widget with dark theme
        self.log_text = tk.Text(log_frame, height=18, width=90,
                              bg="#1e1e1e", #black background
                              fg="#00ff00",  # green text
                              font=('Arial', 10), insertbackground='#00ff00',# green blinking cursor
                              selectbackground='#404040',#dark gray highlight
                              wrap='word')
        
        # Scrollbar for log
        scrollbar = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack log and scrollbar
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add initial welcome message
        welcome_msg = """
        Training System Ready 

        System initialized successfully!
        Configure parameters above and click 'Start Training'

        """
        self.log_text.insert(tk.END, welcome_msg)

    def create_control_buttons(self, parent):        
        # Button container with centered layout
        btn_container = tk.Frame(parent, bg='#2b2b2b')
        btn_container.pack(expand=True)
        
        # Start button
        self.start_btn = tk.Button(btn_container,
                                 text="START TRAINING",command=self.start_training,
                                bg='#4CAF50',fg='white',font=('Arial', 12, 'bold'),
                                 relief='raised',bd=3,padx=20,pady=10,cursor='hand2')
        
        # Stop button
        self.stop_btn = tk.Button(btn_container,text="STOP TRAINING",command=self.stop_training,
                                bg='#f44336',fg='white',font=('Arial', 12, 'bold'),relief='raised',
                                bd=3,padx=20,pady=10,cursor='hand2',state="disabled")
        
        # Pack buttons with spacing
        self.start_btn.pack(side='left', padx=(0, 20))
        self.stop_btn.pack(side='left')

        self.add_button_hover_effects()

    def add_button_hover_effects(self):
        """Add hover effects to buttons."""
        def on_enter_start(e):# when hover on start
            if self.start_btn['state'] != 'disabled':
                self.start_btn.config(bg='#45a049')
        
        def on_leave_start(e):# when leave start
            if self.start_btn['state'] != 'disabled':
                self.start_btn.config(bg='#4CAF50')
        
        def on_enter_stop(e):# when hover stop
            if self.stop_btn['state'] != 'disabled':
                self.stop_btn.config(bg='#da190b')
        
        def on_leave_stop(e):# when leave stop button
            if self.stop_btn['state'] != 'disabled':
                self.stop_btn.config(bg='#f44336')
        
        # calling function specified if any of the conditions are met
        self.start_btn.bind("<Enter>", on_enter_start)
        self.start_btn.bind("<Leave>", on_leave_start)
        self.stop_btn.bind("<Enter>", on_enter_stop)
        self.stop_btn.bind("<Leave>", on_leave_stop)
    '''
    def get_latest_log_dir(self):
        """Finds the path to the most recently created log directory for the current product."""
        product_name = self.entries["entry_train"].get()
        log_base_dir = os.path.join("log", product_name)
        
        if not os.path.exists(log_base_dir):
            return None

        list_of_dirs = glob.glob(os.path.join(log_base_dir, '*'))
        if not list_of_dirs:
            return None
        
        latest_dir = max(list_of_dirs, key=os.path.getmtime)
        return latest_dir
    '''
    def start_training(self):
        """Start the training process via subprocess with validation."""
        #debugging lines
        import sys, os, jax
        self.log_text.insert(tk.END, f"Python: {sys.version}\n")
        self.log_text.insert(tk.END, f"Working dir: {os.getcwd()}\n") 
        self.log_text.insert(tk.END, f"JAX version: {jax.__version__}\n")
        self.log_text.insert(tk.END, f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}\n")
        self.log_text.insert(tk.END, "═" * 50 + "\n")


        # Clear log and add starting message
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "INITIATING TRAINING SEQUENCE\n")
        self.log_text.insert(tk.END, "═" * 50 + "\n")
        self.log_text.insert(tk.END, "Validating parameters...\n\n")
        
        # Input validation
        try:
            int(self.entries["entry_iter"].get())
            float(self.entries["entry_atol"].get())
            float(self.entries["entry_rtol"].get())
            int(self.entries["entry_mxstep"].get())
            float(self.entries["entry_lr"].get())
            int(self.entries["entry_gen"].get())
            
            self.log_text.insert(tk.END, "Parameter validation successful!\n")
            
        except ValueError:
            self.log_text.insert(tk.END, "ERROR: Invalid numeric parameters!\n")
            messagebox.showerror("Invalid Input", "Please enter valid numbers for numeric fields.")
            return
       
        # Construct the command for running the code
        cmd = [
            "python", "-u", "-m", "oven.parameter_estimation",
            f"--iter={self.entries['entry_iter'].get()}",
            f"--atol={self.entries['entry_atol'].get()}",
            f"--rtol={self.entries['entry_rtol'].get()}",
            f"--mxstep={self.entries['entry_mxstep'].get()}",
            f"--lr={self.entries['entry_lr'].get()}",
            f"--gen={self.entries['entry_gen'].get()}",
            f"--load_dir={self.entries['entry_load_dir'].get()}",
            f"--products_train={self.entries['entry_train'].get()}",
            f"--products_test={self.entries['entry_test'].get()}",
            f"--optimize={self.vars['optimize_var'].get()}"
        ]
        
        print("DEBUG CMD:", " ".join(cmd))
        try:
            # Add environment variable for unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env, 
                universal_newlines=True
            )
             # Debugging line:
            self.log_text.insert(tk.END, f"Process started with PID: {self.process.pid}\n")
            self.log_text.insert(tk.END, f"Command: {' '.join(cmd)}\n\n")

            
            # Update button states with visual feedback
            self.start_btn.config(state="disabled", bg='#cccccc')
            self.stop_btn.config(state="normal", bg='#f44336')
            
            # Start a thread to read stdout
            threading.Thread(target=self.stream_logs, daemon=True).start()
            self.parent.after(100, self.poll_log_queue)
            # Start log file monitoring
            threading.Thread(target=self.monitor_log_files, daemon=True).start()

        except FileNotFoundError:
            self.log_text.insert(tk.END, "FATAL ERROR: Command not found!\n")
            messagebox.showerror("Error", "Command execution failed. Check the system.")
            self.reset_buttons()
        except Exception as e:
            self.log_text.insert(tk.END, f"FATAL ERROR: {str(e)}\n")
            messagebox.showerror("Error", f"Failed to start training:\n{e}")
            self.reset_buttons()

    def stream_logs(self):
        """Read stdout and stderr in background threads."""
        def enqueue_output(stream):
            try:
                while True:
                    line = stream.readline()
                    if not line:  # EOF
                        break
                    self.log_queue.put(line)
            except Exception as e:
                self.log_queue.put(f"ERROR reading stream: {str(e)}\n")
            finally:
                stream.close()

        if not self.process:
            return

        # Only need one thread since we're combining stdout and stderr
        threading.Thread(target=enqueue_output, args=(self.process.stdout,), daemon=True).start()
        '''
    def monitor_log_files(self):
        """Monitor the log files created by the parameter estimation script"""
        process_start_time = time.time()
        time.sleep(5)  # Wait for logs to start
        
        product_name = self.entries["entry_train"].get()
        log_base_dir = os.path.join("log", product_name)
        
        # Wait for log directory to be created
        for _ in range(60):
            if os.path.exists(log_base_dir):
                break 
            time.sleep(3)
        
        if os.path.exists(log_base_dir):
            # Only look for directories created AFTER process start
            subdirs = []
            for d in os.listdir(log_base_dir):
                dir_path = os.path.join(log_base_dir,d)
                if os.path.isdir(dir_path):
                    dir_creation_time = os.path.getmtime(dir_path)

                    #Only consider directories created after the process start
                    if dir_creation_time > process_start_time:
                        subdirs.append(dir_path)
            
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                self.log_text.insert(tk.END, f"Found log directory: {latest_dir}\n")
                self.tail_existing_logs(latest_dir)  # Look in the subdirectory!
            else:
                self.log_text.insert(tk.END, f" No new directories found yet. Process started at: {time.ctime(process_start_time)}\n")
                self.log_text.insert(tk.END, f"Looking in: {log_base_dir}\n")
            
            # Debug: show all directories and their times
                self.log_text.insert(tk.END, "Existing directories:\n")
                for d in os.listdir(log_base_dir):
                    dir_path = os.path.join(log_base_dir, d)
                    if os.path.isdir(dir_path):
                        dir_time = os.path.getmtime(dir_path)
                        self.log_text.insert(tk.END, f"  - {d} (created: {time.ctime(dir_time)})\n")
        '''
    
    def monitor_log_files(self):
        """Smart monitoring that finds the latest log directory without hardcoded timing"""
        product_name = self.entries["entry_train"].get()
        log_base_dir = os.path.join("log", product_name)
        
        def smart_monitor():
            max_wait_time = 120  # Maximum 2 minutes total
            check_interval = 2   # Check every 2 seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Check if process died
                if self.process and self.process.poll() is not None:
                    self.log_queue.put("❌ Training process terminated unexpectedly\n")
                    return
                    
                # Check if log directory exists
                if os.path.exists(log_base_dir):
                    # Find ALL directories and get the newest one
                    all_dirs = []
                    try:
                        for item in os.listdir(log_base_dir):
                            item_path = os.path.join(log_base_dir, item)
                            if os.path.isdir(item_path):
                                all_dirs.append((item_path, os.path.getmtime(item_path)))
                        
                        if all_dirs:
                            # Sort by modification time (newest first)
                            all_dirs.sort(key=lambda x: x[1], reverse=True)
                            latest_dir, latest_time = all_dirs[0]
                            
                            # Check if this directory was created AFTER we started training
                            training_start_time = start_time - 10  # Give 10 second buffer
                            if latest_time > training_start_time:
                                self.log_queue.put(f"✅ Found new log directory: {os.path.basename(latest_dir)}\n")
                                self.log_queue.put(f"📅 Created: {time.ctime(latest_time)}\n")
                                self.tail_existing_logs(latest_dir)
                                return
                            else:
                                # Found directories, but they're old - training might be slow
                                elapsed = time.time() - start_time
                                self.log_queue.put(f"⏳ Found existing directories, waiting for new ones... ({elapsed:.0f}s)\n")
                        else:
                            # Directory exists but empty
                            elapsed = time.time() - start_time
                            self.log_queue.put(f"⏳ Log directory exists but empty... ({elapsed:.0f}s)\n")
                            
                    except Exception as e:
                        self.log_queue.put(f"⚠️ Error scanning directory: {e}\n")
                else:
                    # Directory doesn't exist yet
                    elapsed = time.time() - start_time
                    self.log_queue.put(f"⏳ Waiting for log directory... ({elapsed:.0f}s)\n")
                
                time.sleep(check_interval)
            
            # If we get here, timeout occurred
            self.log_queue.put("⏰ Timeout: No new log directories found after 2 minutes\n")
            self.log_queue.put("💡 Training might be running slowly or not creating logs\n")
            
            # Try to use ANY directory as fallback
            if os.path.exists(log_base_dir):
                all_dirs = []
                for item in os.listdir(log_base_dir):
                    item_path = os.path.join(log_base_dir, item)
                    if os.path.isdir(item_path):
                        all_dirs.append((item_path, os.path.getmtime(item_path)))
                
                if all_dirs:
                    all_dirs.sort(key=lambda x: x[1], reverse=True)
                    fallback_dir = all_dirs[0][0]
                    self.log_queue.put(f"🔄 Using fallback directory: {os.path.basename(fallback_dir)}\n")
                    self.tail_existing_logs(fallback_dir)
        
        threading.Thread(target=smart_monitor, daemon=True).start()
        '''
    def tail_existing_logs(self, log_dir):
        """Read and display existing log files"""
        try:
            # Look for log files in the directory
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    # Look for stats files or text files
                    if ('stats' in file.lower() or 'solver' in file.lower() or 
                        file.endswith('.txt')):
                        
                        file_path = os.path.join(root, file)
                        self.log_text.insert(tk.END, f"📄 Monitoring: {file}\n")
                        
                        # Read existing content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if content:
                                self.log_text.insert(tk.END, f"\n=== {file} ===\n{content}\n")
                        
                        # Start continuous monitoring
                        self.start_continuous_monitoring(file_path)
                        
            self.log_text.see(tk.END)
            
        except Exception as e:
            self.log_text.insert(tk.END, f"Note: Could not read logs: {e}\n")
    
        ''''''   
        '''
    def tail_existing_logs(self, log_dir):
        """Read and display existing log files with directory name"""
        try:
            dir_name = os.path.basename(log_dir)
            
            # ✅ Use log_queue to include directory info
            self.log_queue.put(f"📂 Scanning directory: {dir_name}\n")
            
            # Look for log files in the directory
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    # Look for stats files or text files
                    if ('stats' in file.lower() or 'solver' in file.lower() or 
                        file.endswith('.txt')):
                        
                        file_path = os.path.join(root, file)
                        
                        # ✅ Use log_queue to include directory name
                        self.log_queue.put(f"📄 Monitoring [{dir_name}]: {file}\n")
                        
                        # Read existing content
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if content:
                                    self.log_queue.put(f"\n=== {file} (from {dir_name}) ===\n{content}\n")
                        except Exception as e:
                            self.log_queue.put(f"⚠️ Could not read {file}: {e}\n")
                        
                        # Start continuous monitoring - pass directory name
                        self.start_continuous_monitoring(file_path, dir_name)
                            
            self.log_queue.put(f"✅ Finished scanning directory: {dir_name}\n")
                
        except Exception as e:
            self.log_queue.put(f"❌ Error reading logs from {log_dir}: {e}\n") 
            '''
    def start_continuous_monitoring(self, log_path):
        """Continuously monitor a log file for new content"""
        def monitor():
            last_size = 0
            while self.process and self.process.poll() is None:
                try:
                    if os.path.exists(log_path):
                        current_size = os.path.getsize(log_path)
                        if current_size > last_size:
                            # Read new content
                            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                f.seek(last_size)
                                new_content = f.read()
                                if new_content.strip():
                                    # Add to queue so it appears in GUI with timestamp
                                    self.log_queue.put(f"[LOG] {new_content}")
                            last_size = current_size
                        elif current_size < last_size:
                            # File was reset/truncated
                            last_size = 0
                    else:
                        # File doesn't exist yet, wait for it
                        pass
                except Exception as e:
                    # Ignore errors and keep trying
                    pass
                
                time.sleep(1)  # Check every second
        
        # Start monitoring in background thread
        threading.Thread(target=monitor, daemon=True).start()
            '''
    def start_continuous_monitoring(self, log_path, dir_name=None):
        """Continuously monitor a log file for new content with directory info"""
        if dir_name is None:
            # Extract directory name from path if not provided
            dir_name = os.path.basename(os.path.dirname(log_path))
        
        file_name = os.path.basename(log_path)
        
        def monitor():
            last_size = 0
            while self.process and self.process.poll() is None:
                try:
                    if os.path.exists(log_path):
                        current_size = os.path.getsize(log_path)
                        if current_size > last_size:
                            # Read new content
                            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                f.seek(last_size)
                                new_content = f.read()
                                if new_content.strip():
                                    # ✅ Include directory name in log output
                                    self.log_queue.put(f"[{dir_name}/{file_name}] {new_content}")
                            last_size = current_size
                        elif current_size < last_size:
                            # File was reset/truncated
                            self.log_queue.put(f"🔄 File {dir_name}/{file_name} was reset\n")
                            last_size = 0
                    else:
                        # File doesn't exist yet, wait for it
                        pass
                except Exception as e:
                    # Ignore errors and keep trying
                    pass
                
                time.sleep(1)  # Check every second
            
            self.log_queue.put(f"⏹️ Stopped monitoring: {dir_name}/{file_name}\n")
        
        # Start monitoring in background thread
        threading.Thread(target=monitor, daemon=True).start()
    def poll_log_queue(self):
        """Check the queue for new log data and update the Text widget with colors."""
        try:
            while True:
                line = self.log_queue.get_nowait()
                
                # Add timestamp to each line
                timestamp = time.strftime("[%H:%M:%S] ")
                formatted_line = timestamp + line
                
                # Color code different types of messages
                if any(error_word in line.lower() for error_word in ["error", "failed", "exception", "traceback"]):
                    self.log_text.insert(tk.END, formatted_line)
                    self.log_text.tag_add("error", "end-1c linestart", "end-1c")
                    self.log_text.tag_config("error", foreground="#ff6b6b")
                elif "warning" in line.lower():
                    self.log_text.insert(tk.END, formatted_line)
                    self.log_text.tag_add("warning", "end-1c linestart", "end-1c")
                    self.log_text.tag_config("warning", foreground="#ffd93d")
                elif any(epoch_word in line.lower() for epoch_word in ["epoch", "iteration", "step"]):
                    self.log_text.insert(tk.END, formatted_line)
                    self.log_text.tag_add("epoch", "end-1c linestart", "end-1c")
                    self.log_text.tag_config("epoch", foreground="#00ccff")
                elif any(success_word in line.lower() for success_word in ["success", "completed", "finished", "done"]):
                    self.log_text.insert(tk.END, formatted_line)
                    self.log_text.tag_add("success", "end-1c linestart", "end-1c")
                    self.log_text.tag_config("success", foreground="#6bcf7f")
                else:
                    self.log_text.insert(tk.END, formatted_line)
                
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        
        # Check if process is still running
        if self.process:
            return_code = self.process.poll()
            if return_code is None:
                # Process still running, continue polling
                self.parent.after(100, self.poll_log_queue)
            else:
                # Process finished
                self.reset_buttons()
                if return_code == 0:
                    success_msg = "✅ TRAINING COMPLETED SUCCESSFULLY! ✅\n"
                    self.log_text.insert(tk.END, success_msg)
                    self.log_text.tag_add("success", "end-4c linestart", "end-1c")
                    self.log_text.tag_config("success", foreground="#6bcf7f")
                else:
                    error_msg = f"TRAINING PROCESS TERMINATED WITH ERROR CODE: {return_code} \n"
                    self.log_text.insert(tk.END, error_msg)
                    self.log_text.tag_add("error", "end-4c linestart", "end-1c")
                    self.log_text.tag_config("error", foreground="#ff6b6b")

    def stop_training(self):
        """Stop the training process with visual feedback."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            
            self.log_text.insert(tk.END, "\nTRAINING STOPPED BY USER\n")
            self.reset_buttons()

    def reset_buttons(self):
        """Reset button states after training stops with visual feedback."""
        self.start_btn.config(state="normal", bg='#4CAF50')
        self.stop_btn.config(state="disabled", bg='#cccccc')
        self.process = None

    def cleanup(self):
        """Cleanup on window close."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.parent.winfo_toplevel().destroy()