#!/bin/bash

# AlgoGen Applications Launcher
# This script launches the AlgoGen v6.0 application

echo "Starting AlgoGen Applications..."

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/applications/algogen_v6"

# Check if we're in the right directory
if [ ! -d "applications/algogen_v6" ]; then
    echo "Error: applications/algogen_v6 directory not found!"
    echo "Please run this script from the repo root directory."
    exit 1
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $port is already in use. Please stop the service using port $port first."
        return 1
    fi
    return 0
}

# Function to start AlgoGen v6.0
start_algogen_v6() {
    echo "Starting AlgoGen v6.0..."
    
    cd applications/algogen_v6
    
    # Check if required files exist
    if [ ! -f "web/app.py" ]; then
        echo "Error: web/app.py not found!"
        return 1
    fi
    
    # Check if port 5000 is available
    if ! check_port 5000; then
        echo "Trying alternative port 5001..."
        if ! check_port 5001; then
            echo "Both ports 5000 and 5001 are in use. Please free up a port."
            return 1
        fi
        export FLASK_RUN_PORT=5001
        echo "AlgoGen v6.0 will start on port 5001"
    else
        export FLASK_RUN_PORT=5000
        echo "AlgoGen v6.0 will start on port 5000"
    fi
    
    # Set Flask environment
    export FLASK_APP=web/app.py
    export FLASK_ENV=development
    
    echo "Starting Flask application..."
    echo "Access the application at: http://localhost:$FLASK_RUN_PORT"
    echo "Press Ctrl+C to stop the application"
    
    # Start the application
    python web/app.py
}

# Function to show help
show_help() {
    echo "AlgoGen Applications Launcher"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  v6, algogen-v6    Start AlgoGen v6.0 application"
    echo "  help, -h, --help  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 v6              # Start AlgoGen v6.0"
    echo "  $0 algogen-v6      # Start AlgoGen v6.0"
    echo ""
}

# Main script logic
case "${1:-v6}" in
    "v6"|"algogen-v6")
        start_algogen_v6
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 