#!/bin/bash

# SentinelFS AI - Monitoring Setup Script
# Sets up Prometheus and Grafana for production monitoring

set -e

echo "🚀 Setting up SentinelFS AI Production Monitoring..."

# Configuration
PROMETHEUS_VERSION="2.45.0"
GRAFANA_VERSION="10.2.0"
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
API_METRICS_PORT=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is available
check_docker() {
    if command -v docker &> /dev/null; then
        print_status "Docker found"
        return 0
    else
        print_warning "Docker not found. Installing Prometheus and Grafana locally..."
        return 1
    fi
}

# Setup Prometheus configuration
setup_prometheus_config() {
    print_status "Setting up Prometheus configuration..."

    cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'sentinelfs_api'
    static_configs:
      - targets: ['localhost:$API_METRICS_PORT']
    metrics_path: '/prometheus/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:$PROMETHEUS_PORT']
EOF

    print_status "Prometheus configuration created"
}

# Setup Grafana provisioning
setup_grafana_provisioning() {
    print_status "Setting up Grafana provisioning..."

    mkdir -p grafana/provisioning/datasources
    mkdir -p grafana/provisioning/dashboards

    # Create datasource configuration
    cat > grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:$PROMETHEUS_PORT
    isDefault: true
    editable: true
EOF

    # Create dashboard provisioning
    cat > grafana/provisioning/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'sentinelfs'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    print_status "Grafana provisioning configured"
}

# Setup using Docker Compose
setup_docker_compose() {
    print_status "Setting up Docker Compose configuration..."

    cat > docker-compose.monitoring.yml << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v$PROMETHEUS_VERSION
    container_name: sentinelfs_prometheus
    ports:
      - "$PROMETHEUS_PORT:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:$GRAFANA_VERSION
    container_name: sentinelfs_grafana
    ports:
      - "$GRAFANA_PORT:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./sentinelfs_ai/monitoring/grafana/dashboard.json:/var/lib/grafana/dashboards/sentinelfs.json
    networks:
      - monitoring
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
EOF

    print_status "Docker Compose configuration created"
}

# Install Prometheus locally
install_prometheus_local() {
    print_status "Installing Prometheus locally..."

    # Download and extract Prometheus
    wget -q https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz
    tar xfz prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz
    cd prometheus-$PROMETHEUS_VERSION.linux-amd64

    # Create systemd service
    sudo tee /etc/systemd/system/prometheus.service > /dev/null << EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \\
  --config.file=/etc/prometheus/prometheus.yml \\
  --storage.tsdb.path=/var/lib/prometheus \\
  --web.console.templates=/etc/prometheus/consoles \\
  --web.console.libraries=/etc/prometheus/console_libraries \\
  --storage.tsdb.retention.time=200h \\
  --web.enable-lifecycle

[Install]
WantedBy=multi-user.target
EOF

    # Setup user and directories
    sudo useradd --no-create-home --shell /bin/false prometheus
    sudo mkdir -p /etc/prometheus /var/lib/prometheus
    sudo chown prometheus:prometheus /etc/prometheus /var/lib/prometheus

    # Copy files
    sudo cp prometheus /usr/local/bin/
    sudo cp promtool /usr/local/bin/
    sudo cp -r consoles /etc/prometheus/
    sudo cp -r console_libraries /etc/prometheus/
    sudo cp ../prometheus.yml /etc/prometheus/

    # Start service
    sudo systemctl daemon-reload
    sudo systemctl enable prometheus
    sudo systemctl start prometheus

    cd ..
    rm -rf prometheus-$PROMETHEUS_VERSION.linux-amd64*

    print_status "Prometheus installed and started"
}

# Install Grafana locally
install_grafana_local() {
    print_status "Installing Grafana locally..."

    # Install Grafana
    wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
    echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
    sudo apt-get update
    sudo apt-get install -y grafana

    # Copy provisioning files
    sudo cp -r grafana/provisioning/* /etc/grafana/provisioning/
    sudo cp sentinelfs_ai/monitoring/grafana/dashboard.json /var/lib/grafana/dashboards/

    # Start Grafana
    sudo systemctl enable grafana-server
    sudo systemctl start grafana-server

    print_status "Grafana installed and started"
}

# Import dashboard via API
import_grafana_dashboard() {
    print_status "Importing SentinelFS dashboard..."

    # Wait for Grafana to be ready
    sleep 10

    # Import dashboard
    curl -X POST -H "Content-Type: application/json" \
         -d @sentinelfs_ai/monitoring/grafana/dashboard.json \
         http://admin:admin@localhost:$GRAFANA_PORT/api/dashboards/db || true

    print_status "Dashboard imported"
}

# Main setup function
main() {
    print_status "Starting SentinelFS AI monitoring setup..."

    # Create necessary directories
    mkdir -p grafana/provisioning/datasources grafana/provisioning/dashboards

    # Setup configurations
    setup_prometheus_config
    setup_grafana_provisioning

    if check_docker; then
        # Docker setup
        setup_docker_compose
        print_status "Starting monitoring stack with Docker Compose..."
        docker-compose -f docker-compose.monitoring.yml up -d

        # Import dashboard
        import_grafana_dashboard
    else
        # Local installation
        install_prometheus_local
        install_grafana_local
        import_grafana_dashboard
    fi

    print_status "🎉 Monitoring setup complete!"
    echo ""
    echo "Access your monitoring dashboards:"
    echo "- Prometheus: http://localhost:$PROMETHEUS_PORT"
    echo "- Grafana: http://localhost:$GRAFANA_PORT (admin/admin)"
    echo "- SentinelFS API: http://localhost:$API_METRICS_PORT/docs"
    echo ""
    echo "Grafana Dashboard: 'SentinelFS AI - Production Monitoring'"
    echo ""
    print_warning "Remember to change the default Grafana password!"
}

# Run main function
main "$@"