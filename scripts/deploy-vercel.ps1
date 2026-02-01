# CCRD Vercel Deployment Setup Script (Windows PowerShell)
# This script helps configure CCRD for Vercel deployment

Write-Host "=========================================="
Write-Host "CCRD Vercel Deployment Setup" -ForegroundColor Green
Write-Host "=========================================="
Write-Host ""

# Check if git repo exists
if (-not (Test-Path ".git")) {
    Write-Host "Error: Not a git repository. Please run 'git init' first." -ForegroundColor Red
    exit 1
}

# Check if Vercel CLI is installed
$vercelExists = $null -ne (Get-Command vercel -ErrorAction SilentlyContinue)
if (-not $vercelExists) {
    Write-Host "Installing Vercel CLI..."
    npm install -g vercel
}

Write-Host "Step 1: Configure Backend API URL" -ForegroundColor Cyan
Write-Host "=================================="
$backendUrl = Read-Host "Enter your backend API URL (e.g., https://ccrd-api.onrender.com)"

Write-Host ""
Write-Host "Step 2: Create Vercel project" -ForegroundColor Cyan
Write-Host "============================="
Write-Host "Running 'vercel' command to create/link project..."
& vercel

Write-Host ""
Write-Host "Step 3: Set environment variables in Vercel" -ForegroundColor Cyan
Write-Host "==========================================="
Write-Host "Setting NEXT_PUBLIC_API_URL=$backendUrl"
& vercel env add NEXT_PUBLIC_API_URL

Write-Host ""
Write-Host "Step 4: Configure vercel.json" -ForegroundColor Cyan
Write-Host "============================="

$vercelConfig = @"
{
  "buildCommand": "npm run build || echo 'No Next.js build'",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "env": {
    "NEXT_PUBLIC_API_URL": {
      "description": "Backend API URL",
      "default": "$backendUrl",
      "required": false
    }
  },
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "$backendUrl/api/:path*"
    }
  ],
  "redirects": [
    {
      "source": "/docs",
      "destination": "$backendUrl/docs",
      "permanent": false
    }
  ]
}
"@

Set-Content -Path "vercel.json" -Value $vercelConfig
Write-Host "✓ vercel.json created" -ForegroundColor Green

Write-Host ""
Write-Host "Step 5: Deploy to Vercel" -ForegroundColor Cyan
Write-Host "======================="
$deploy = Read-Host "Deploy now? (y/n)"

if ($deploy -eq 'y' -or $deploy -eq 'Y') {
    & vercel --prod
    Write-Host "✓ Deployment complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your frontend is now live on Vercel!"
    Write-Host "Check your Vercel dashboard for the deployment URL."
} else {
    Write-Host "Skipped deployment. Run 'vercel --prod' when ready."
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "==========="
Write-Host "1. Update CORS on backend:"
Write-Host "   - Add your Vercel URL to backend CORS allowed origins"
Write-Host "2. Test API connection:"
Write-Host "   - Open browser DevTools and test API calls"
Write-Host "3. Configure custom domain (optional):"
Write-Host "   - Add in Vercel dashboard under Project Settings"
Write-Host ""
