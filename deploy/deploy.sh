#!/bin/bash
# =============================================================================
# Credibility AI - EC2 Deployment Script
# =============================================================================
# Usage: ./deploy/deploy.sh [command]
# Commands: build, up, down, restart, logs, status, clean
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project name
PROJECT_NAME="credibility-ai"

# Print colored message
print_msg() {
    echo -e "${GREEN}[${PROJECT_NAME}]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if .env file exists
check_env() {
    if [ ! -f ".env" ]; then
        print_error ".env file not found!"
        print_msg "Copy the example file: cp deploy/env.example .env"
        print_msg "Then edit .env with your configuration"
        exit 1
    fi
}

# Build Docker images
build() {
    print_msg "Building Docker images..."
    docker-compose build --no-cache
    print_msg "Build completed!"
}

# Start all services
up() {
    check_env
    print_msg "Starting all services..."
    docker-compose up -d
    print_msg "Services started!"
    print_msg "API available at: http://localhost:${API_PORT:-8000}"
    status
}

# Stop all services
down() {
    print_msg "Stopping all services..."
    docker-compose down
    print_msg "Services stopped!"
}

# Restart all services
restart() {
    print_msg "Restarting all services..."
    docker-compose restart
    print_msg "Services restarted!"
    status
}

# View logs
logs() {
    SERVICE=${1:-""}
    if [ -z "$SERVICE" ]; then
        docker-compose logs -f --tail=100
    else
        docker-compose logs -f --tail=100 "$SERVICE"
    fi
}

# Check status of services
status() {
    print_msg "Service Status:"
    echo ""
    docker-compose ps
    echo ""
    print_msg "Health Check:"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
}

# Clean up Docker resources
clean() {
    print_warning "This will remove all stopped containers, unused networks, and dangling images."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_msg "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_msg "Cleanup completed!"
    fi
}

# Pull latest images
pull() {
    print_msg "Pulling latest images..."
    docker-compose pull
    print_msg "Pull completed!"
}

# Scale a service
scale() {
    SERVICE=$1
    COUNT=$2
    if [ -z "$SERVICE" ] || [ -z "$COUNT" ]; then
        print_error "Usage: ./deploy.sh scale <service> <count>"
        print_msg "Example: ./deploy.sh scale sqs-worker 3"
        exit 1
    fi
    print_msg "Scaling $SERVICE to $COUNT instances..."
    docker-compose up -d --scale "$SERVICE=$COUNT"
    print_msg "Scaling completed!"
}

# Show help
help() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           Credibility AI - Deployment Script                  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: ./deploy/deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build     - Build Docker images"
    echo "  up        - Start all services"
    echo "  down      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  logs      - View logs (optional: specify service name)"
    echo "  status    - Check status of services"
    echo "  clean     - Clean up Docker resources"
    echo "  pull      - Pull latest images"
    echo "  scale     - Scale a service (e.g., scale sqs-worker 3)"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy/deploy.sh build"
    echo "  ./deploy/deploy.sh up"
    echo "  ./deploy/deploy.sh logs api"
    echo "  ./deploy/deploy.sh scale sqs-worker 2"
    echo ""
}

# Main
case "${1:-help}" in
    build)
        build
        ;;
    up)
        up
        ;;
    down)
        down
        ;;
    restart)
        restart
        ;;
    logs)
        logs "$2"
        ;;
    status)
        status
        ;;
    clean)
        clean
        ;;
    pull)
        pull
        ;;
    scale)
        scale "$2" "$3"
        ;;
    help|*)
        help
        ;;
esac

