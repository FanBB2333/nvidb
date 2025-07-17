#!/usr/bin/env python3
"""
NVIDB Interactive GPU Monitoring Example

This example demonstrates how to use the new interactive keyboard listening functionality
"""

from nvidb.src.connection import NviClientPool, LocalClient

def main():
    """Main function - Demonstrates interactive GPU monitoring"""
    print("🚀 NVIDB Interactive GPU Monitoring Example")
    print("=" * 50)
    
    try:
        # Create client pool (using only local client as example)
        # In actual usage, you can pass in remote server list
        pool = NviClientPool(server_list=None)
        
        # Start interactive monitoring
        # Users can:
        #   - Type 'q' to exit
        #   - Type 'h' to show help
        #   - Press Enter to refresh immediately
        #   - Press Ctrl+C to force exit
        pool.print_refresh()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("👋 Program finished")

if __name__ == "__main__":
    main()
