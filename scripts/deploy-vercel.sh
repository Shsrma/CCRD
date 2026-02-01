#!/bin/bash

# CCRD Vercel Deployment Setup Script
# This script helps configure CCRD for Vercel deployment

set -e

echo "=========================================="
echo "CCRD Vercel Deployment Setup"
echo "=========================================="
echo ""

# Check if git repo exists
if [ ! -d ".git" ]; then
    echo "Error: Not a git repository. Please run 'git init' first."
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Installing Vercel CLI..."
    npm install -g vercel
fi

echo "Step 1: Configure Backend API URL"
echo "=================================="
read -p "Enter your backend API URL (e.g., https://ccrd-api.onrender.com): " BACKEND_URL

echo ""
echo "Step 2: Create Vercel project"
echo "============================="
echo "Running 'vercel' command to create/link project..."
vercel

echo ""
echo "Step 3: Set environment variables in Vercel"
echo "==========================================="
echo "Setting NEXT_PUBLIC_API_URL=$BACKEND_URL"
vercel env add NEXT_PUBLIC_API_URL

echo ""
echo "Step 4: Configure vercel.json"
echo "============================="
cat > vercel.json << EOF
{
  "buildCommand": "npm run build || echo 'No Next.js build'",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "env": {
    "NEXT_PUBLIC_API_URL": {
      "description": "Backend API URL",
      "default": "$BACKEND_URL",
      "required": false
    }
  },
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "$BACKEND_URL/api/:path*"
    }
  ],
  "redirects": [
    {
      "source": "/docs",
      "destination": "$BACKEND_URL/docs",
      "permanent": false
    }
  ]
}
EOF

echo "✓ vercel.json created"

echo ""
echo "Step 5: Deploy to Vercel"
echo "======================="
read -p "Deploy now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    vercel --prod
    echo "✓ Deployment complete!"
    echo ""
    echo "Your frontend is now live on Vercel!"
    echo "Check your Vercel dashboard for the deployment URL."
else
    echo "Skipped deployment. Run 'vercel --prod' when ready."
fi

echo ""
echo "Next steps:"
echo "==========="
echo "1. Update CORS on backend:"
echo "   - Add your Vercel URL to backend CORS allowed origins"
echo "2. Test API connection:"
echo "   - Open browser DevTools and test API calls"
echo "3. Configure custom domain (optional):"
echo "   - Add in Vercel dashboard under Project Settings"
echo ""
