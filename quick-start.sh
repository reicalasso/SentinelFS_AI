# Quick Start Deployment Script
#!/bin/bash

set -e

echo "🚀 SentinelZer0 Quick Start Deployment"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before proceeding."
    echo "   Especially change:"
    echo "   - API_SECRET_KEY"
    echo "   - GRAFANA_ADMIN_PASSWORD"
    echo "   - Database credentials (if using)"
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models/production data logs monitoring/grafana/dashboards monitoring/grafana/datasources

# Build Docker images
echo "🔨 Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
services=("sentinelzer0-api" "sentinelzer0-redis" "sentinelzer0-prometheus" "sentinelzer0-grafana")
for service in "${services[@]}"; do
    if docker ps | grep -q $service; then
        echo "✅ $service is running"
    else
        echo "❌ $service is not running"
    fi
done

# Display access information
echo ""
echo "======================================"
echo "🎉 SentinelZer0 is now running!"
echo "======================================"
echo ""
echo "📊 Access Points:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Grafana: http://localhost:3000 (admin/sentinelzer0)"
echo "  - Prometheus: http://localhost:9091"
echo "  - MLflow: http://localhost:5000"
echo ""
echo "🔍 Useful Commands:"
echo "  - View logs: docker-compose logs -f sentinelzer0"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - View status: docker-compose ps"
echo ""
echo "📚 Next Steps:"
echo "  1. Visit http://localhost:8000/docs to explore the API"
echo "  2. Configure Grafana dashboards at http://localhost:3000"
echo "  3. Check the logs: docker-compose logs -f"
echo "  4. Read the documentation in README_MAIN.md"
echo ""

# Test API health
echo "🧪 Testing API health..."
sleep 5
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is responding"
else
    echo "⚠️  API is not responding yet. It may need more time to start."
    echo "   Check logs with: docker-compose logs -f sentinelzer0"
fi

echo ""
echo "======================================"
echo "Happy threat hunting! 🛡️"
echo "======================================"
