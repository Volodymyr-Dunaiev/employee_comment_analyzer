"""
Launcher for the standalone Comments Classifier executable.
This starts the Streamlit app and opens it in the default browser.
"""
import os
import sys
import webbrowser
import subprocess
import time
import threading

def open_browser_delayed(url, delay=3):
    """Open browser after a delay to let Streamlit start."""
    time.sleep(delay)
    webbrowser.open(url)

def main():
    # Set working directory to exe location
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(app_dir)
    
    # Set environment variables
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    print("=" * 60)
    print("Ukrainian Comments Classifier")
    print("=" * 60)
    print(f"Working directory: {app_dir}")
    print("Starting application...")
    print()
    
    # Build streamlit command
    streamlit_script = os.path.join(app_dir, 'src', 'ui', 'app_ui.py')
    port = 8501
    url = f'http://localhost:{port}'
    
    try:
        # Always spawn Streamlit as subprocess (works for both frozen and script)
        if getattr(sys, 'frozen', False):
            # When frozen, need to use python interpreter bundled with exe
            python_exe = sys.executable
            cmd = [
                python_exe,
                '-m', 'streamlit',
                'run',
                streamlit_script,
                f'--server.port={port}',
                '--server.headless=true',
                '--browser.gatherUsageStats=false'
            ]
        else:
            cmd = [
                sys.executable,
                '-m', 'streamlit',
                'run',
                streamlit_script,
                f'--server.port={port}',
                '--server.headless=true',
                '--browser.gatherUsageStats=false'
            ]
        
        # Start Streamlit in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        # Open browser in separate thread after delay
        browser_thread = threading.Thread(target=open_browser_delayed, args=(url, 3))
        browser_thread.daemon = True
        browser_thread.start()
        
        print("Application started successfully!")
        print(f"Opening in browser: {url}")
        print()
        print("Close this window to stop the application.")
        print("=" * 60)
        print()
        
        # Keep window open and wait for user
        try:
            input("Press Enter to exit...")
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        # Cleanup
        process.terminate()
        process.wait(timeout=5)
    
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == '__main__':
    main()
