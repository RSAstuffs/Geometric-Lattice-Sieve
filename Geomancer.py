#!/usr/bin/env python3
"""
Geomancer GUI - A comprehensive graphical interface for the GeometricLLL Factorizer
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import sys
import io
import time
from datetime import datetime

# Import the factorizer
try:
    from Geomancer import GeometricFactorizer
    from Crypto.Util.number import getPrime
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure Geomancer.py and pycryptodome are available")


class OutputRedirector:
    """Redirects stdout to a queue for GUI display"""
    def __init__(self, output_queue):
        self.output_queue = output_queue
        
    def write(self, text):
        self.output_queue.put(text)
        
    def flush(self):
        pass


class GeomancerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔮 Geomancer - Geometric LLL Factorizer")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom colors
        self.bg_color = "#1a1a2e"
        self.fg_color = "#eee"
        self.accent_color = "#e94560"
        self.success_color = "#00ff88"
        self.warning_color = "#ffa500"
        
        self.root.configure(bg=self.bg_color)
        
        # Variables
        self.n_var = tk.StringVar()
        self.lattice_dim_var = tk.StringVar(value="500")
        self.precision_bits_var = tk.StringVar(value="auto")
        self.interval_size_var = tk.StringVar(value="auto")
        self.running = False
        self.factorizer = None
        self.output_queue = queue.Queue()
        
        # Build UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        
        # Start output monitor
        self._monitor_output()
        
    def _create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load N from file...", command=self._load_n_from_file)
        file_menu.add_command(label="Save results...", command=self._save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Generate menu
        gen_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Generate", menu=gen_menu)
        gen_menu.add_command(label="64-bit RSA", command=lambda: self._generate_rsa(32))
        gen_menu.add_command(label="128-bit RSA", command=lambda: self._generate_rsa(64))
        gen_menu.add_command(label="256-bit RSA", command=lambda: self._generate_rsa(128))
        gen_menu.add_command(label="512-bit RSA", command=lambda: self._generate_rsa(256))
        gen_menu.add_command(label="1024-bit RSA", command=lambda: self._generate_rsa(512))
        gen_menu.add_command(label="2048-bit RSA", command=lambda: self._generate_rsa(1024))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="How it works", command=self._show_help)
        
    def _create_main_layout(self):
        """Create the main layout with panels"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Input & Controls
        left_panel = ttk.LabelFrame(main_frame, text="⚙️ Configuration", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # N input
        ttk.Label(left_panel, text="Target N (integer to factor):").pack(anchor=tk.W)
        n_frame = ttk.Frame(left_panel)
        n_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.n_text = scrolledtext.ScrolledText(n_frame, width=40, height=6, wrap=tk.WORD)
        self.n_text.pack(fill=tk.X)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(left_panel, text="Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Lattice dimension
        ttk.Label(params_frame, text="Lattice Dimension:").grid(row=0, column=0, sticky=tk.W, pady=2)
        lattice_entry = ttk.Entry(params_frame, textvariable=self.lattice_dim_var, width=15)
        lattice_entry.grid(row=0, column=1, pady=2, padx=5)
        ttk.Label(params_frame, text="(default: 500)").grid(row=0, column=2, sticky=tk.W)
        
        # Precision bits
        ttk.Label(params_frame, text="Precision Bits:").grid(row=1, column=0, sticky=tk.W, pady=2)
        precision_entry = ttk.Entry(params_frame, textvariable=self.precision_bits_var, width=15)
        precision_entry.grid(row=1, column=1, pady=2, padx=5)
        ttk.Label(params_frame, text="(auto or number)").grid(row=1, column=2, sticky=tk.W)
        
        # Interval size
        ttk.Label(params_frame, text="Sieve Interval:").grid(row=2, column=0, sticky=tk.W, pady=2)
        interval_entry = ttk.Entry(params_frame, textvariable=self.interval_size_var, width=15)
        interval_entry.grid(row=2, column=1, pady=2, padx=5)
        ttk.Label(params_frame, text="(auto or number)").grid(row=2, column=2, sticky=tk.W)
        
        # Buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="🚀 Start Factorization", command=self._start_factorization)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Stop", command=self._stop_factorization, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame, text="🗑️ Clear Output", command=self._clear_output).pack(fill=tk.X, pady=2)
        
        # Quick actions
        quick_frame = ttk.LabelFrame(left_panel, text="Quick Actions", padding=5)
        quick_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(quick_frame, text="📋 Paste from Clipboard", command=self._paste_n).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="📑 Copy Results", command=self._copy_results).pack(fill=tk.X, pady=2)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(left_panel, text="📊 Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_labels = {}
        stats = [("N bits:", "n_bits"), ("Lattice:", "lattice"), ("Relations:", "relations"), 
                 ("Dependencies:", "deps"), ("Time:", "time")]
        
        for i, (label, key) in enumerate(stats):
            ttk.Label(stats_frame, text=label).grid(row=i, column=0, sticky=tk.W)
            self.stats_labels[key] = ttk.Label(stats_frame, text="-")
            self.stats_labels[key].grid(row=i, column=1, sticky=tk.W, padx=10)
        
        # Right panel - Output
        right_panel = ttk.LabelFrame(main_frame, text="📜 Output Log", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, 
                                                      bg="#0d0d0d", fg="#00ff00",
                                                      font=("Consolas", 10),
                                                      insertbackground="#00ff00")
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for colored output
        self.output_text.tag_configure("success", foreground=self.success_color)
        self.output_text.tag_configure("error", foreground=self.accent_color)
        self.output_text.tag_configure("warning", foreground=self.warning_color)
        self.output_text.tag_configure("info", foreground="#00bfff")
        
    def _create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=5)
        
    def _monitor_output(self):
        """Monitor the output queue and update the text widget"""
        try:
            while True:
                text = self.output_queue.get_nowait()
                self._append_output(text)
        except queue.Empty:
            pass
        self.root.after(100, self._monitor_output)
        
    def _append_output(self, text):
        """Append text to output with color coding"""
        self.output_text.config(state=tk.NORMAL)
        
        # Determine tag based on content
        tag = None
        if "[SUCCESS]" in text:
            tag = "success"
        elif "[FAILURE]" in text or "Error" in text:
            tag = "error"
        elif "Warning" in text:
            tag = "warning"
        elif "Running" in text or "Building" in text or "Analyzing" in text:
            tag = "info"
            
        if tag:
            self.output_text.insert(tk.END, text, tag)
        else:
            self.output_text.insert(tk.END, text)
            
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        # Update statistics from output
        self._parse_stats(text)
        
    def _parse_stats(self, text):
        """Parse output text for statistics"""
        if "bit length:" in text:
            try:
                bits = text.split("bit length:")[1].strip().split()[0]
                self.stats_labels["n_bits"].config(text=bits)
            except:
                pass
        if "Lattice dimension:" in text:
            try:
                dim = text.split("Lattice dimension:")[1].strip().split()[0]
                self.stats_labels["lattice"].config(text=dim)
            except:
                pass
        if "Total relations:" in text:
            try:
                rels = text.split("Total relations:")[1].strip().split()[0]
                self.stats_labels["relations"].config(text=rels)
            except:
                pass
        if "independent dependencies" in text:
            try:
                deps = text.split("Found")[1].strip().split()[0]
                self.stats_labels["deps"].config(text=deps)
            except:
                pass
                
    def _start_factorization(self):
        """Start the factorization in a separate thread"""
        n_text = self.n_text.get("1.0", tk.END).strip()
        
        if not n_text:
            messagebox.showerror("Error", "Please enter a number N to factor")
            return
            
        try:
            N = int(n_text.replace(" ", "").replace("\n", ""))
        except ValueError:
            messagebox.showerror("Error", "Invalid number format")
            return
            
        if N < 2:
            messagebox.showerror("Error", "N must be greater than 1")
            return
            
        # Get parameters
        try:
            lattice_dim = int(self.lattice_dim_var.get())
        except:
            lattice_dim = 500
            
        precision = self.precision_bits_var.get()
        precision_bits = None if precision.lower() == "auto" else int(precision)
        
        interval = self.interval_size_var.get()
        interval_size = None if interval.lower() == "auto" else int(interval)
        
        # Update UI
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()
        self.status_var.set("Running factorization...")
        self.start_time = time.time()
        
        # Clear previous stats
        for label in self.stats_labels.values():
            label.config(text="-")
        
        # Start factorization thread
        self.factor_thread = threading.Thread(
            target=self._run_factorization,
            args=(N, lattice_dim, precision_bits, interval_size),
            daemon=True
        )
        self.factor_thread.start()
        
        # Monitor thread
        self._monitor_thread()
        
    def _run_factorization(self, N, lattice_dim, precision_bits, interval_size):
        """Run factorization (called in separate thread)"""
        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = OutputRedirector(self.output_queue)
        
        try:
            print(f"\n{'='*60}")
            print(f"🔮 GEOMANCER - Starting Factorization")
            print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            self.factorizer = GeometricFactorizer(
                N, 
                lattice_dim=lattice_dim,
                precision_bits=precision_bits,
                interval_size=interval_size
            )
            
            if self.running:
                self.factorizer.find_relations()
                
            if self.running:
                self.factorizer.solve_linear_system()
                
            print(f"\n{'='*60}")
            print(f"Factorization complete.")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            
    def _monitor_thread(self):
        """Monitor the factorization thread"""
        if self.factor_thread.is_alive():
            # Update time
            elapsed = time.time() - self.start_time
            self.stats_labels["time"].config(text=f"{elapsed:.1f}s")
            self.root.after(500, self._monitor_thread)
        else:
            self._factorization_complete()
            
    def _factorization_complete(self):
        """Called when factorization is done"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        elapsed = time.time() - self.start_time
        self.status_var.set(f"Complete (took {elapsed:.2f}s)")
        self.stats_labels["time"].config(text=f"{elapsed:.2f}s")
        
    def _stop_factorization(self):
        """Stop the running factorization"""
        self.running = False
        self.status_var.set("Stopping...")
        self.output_queue.put("\n[STOPPED] Factorization cancelled by user.\n")
        
    def _clear_output(self):
        """Clear the output text"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        
    def _paste_n(self):
        """Paste N from clipboard"""
        try:
            text = self.root.clipboard_get()
            self.n_text.delete("1.0", tk.END)
            self.n_text.insert("1.0", text)
        except:
            messagebox.showwarning("Warning", "Nothing to paste from clipboard")
            
    def _copy_results(self):
        """Copy output to clipboard"""
        text = self.output_text.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Results copied to clipboard")
        
    def _generate_rsa(self, bits):
        """Generate a random RSA modulus"""
        self.output_queue.put(f"\nGenerating {bits*2}-bit RSA modulus...\n")
        try:
            p = getPrime(bits)
            q = getPrime(bits)
            N = p * q
            self.n_text.delete("1.0", tk.END)
            self.n_text.insert("1.0", str(N))
            self.output_queue.put(f"Generated N = p * q ({N.bit_length()} bits)\n")
            self.output_queue.put(f"  p = {p}\n")
            self.output_queue.put(f"  q = {q}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate RSA: {e}")
            
    def _load_n_from_file(self):
        """Load N from a file"""
        filename = filedialog.askopenfilename(
            title="Select file containing N",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                self.n_text.delete("1.0", tk.END)
                self.n_text.insert("1.0", content)
                self.status_var.set(f"Loaded N from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                
    def _save_results(self):
        """Save results to a file"""
        filename = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.output_text.get("1.0", tk.END))
                self.status_var.set(f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
                
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About Geomancer",
            "🔮 Geomancer - Geometric LLL Factorizer\n\n"
            "A lattice-based integer factorization tool using\n"
            "the GeometricLLL algorithm.\n\n"
            "Uses Schnorr's lattice construction combined with\n"
            "geometric reduction techniques to find smooth\n"
            "relations and factor composite integers.\n\n"
            "© 2026"
        )
        
    def _show_help(self):
        """Show help dialog"""
        help_text = """
🔮 GEOMANCER - How It Works

1. LATTICE CONSTRUCTION
   Builds a Schnorr-style lattice where short vectors
   correspond to multiplicative relations among small primes.

2. GEOMETRIC REDUCTION  
   Uses the GeometricLLL algorithm to find short vectors
   in the lattice. These represent smooth relations.

3. LINEAR ALGEBRA
   Combines relations using Gaussian elimination over GF(2)
   to find dependencies where exponents sum to even values.

4. FACTOR EXTRACTION
   From dependencies, computes X² ≡ Y² (mod N) and uses
   GCD(X-Y, N) to extract non-trivial factors.

PARAMETERS:
• Lattice Dimension: Larger = more relations but slower
• Precision Bits: Higher = more accurate log approximations
• Sieve Interval: For supplementary relation finding

TIPS:
• Start with lattice_dim = 500 for most numbers
• For very large N, increase lattice dimension
• Need relations > primes for guaranteed solution
"""
        help_window = tk.Toplevel(self.root)
        help_window.title("How Geomancer Works")
        help_window.geometry("500x500")
        
        text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert("1.0", help_text)
        text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = GeomancerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
