# Deploying CCRD to Vercel

## Quick Overview

**Vercel is ideal for the frontend.** For CCRD, you have two deployment strategies:

### Option 1: Frontend on Vercel + Backend on Render (⭐ Recommended)
- **Frontend**: Vercel (fast, free, easy)
- **Backend API**: Render.com or AWS
- **Best for**: Team projects, production

### Option 2: Frontend on Vercel + Backend on Vercel (⚠️ Limited)
- **Frontend**: Vercel
- **Backend**: Vercel Serverless Functions
- **Limitation**: ML model + heavy computations may timeout

This guide covers **Option 1** (Recommended).

---

## Prerequisites

- GitHub repository with CCRD code
- Vercel account (free at [vercel.com](https://vercel.com))
- Render account (for backend) OR other backend hosting
- GitHub CLI or web access

---

## Step 1: Deploy Backend First (Render.com)

Before deploying frontend, deploy the backend API:

```bash
# Follow docs/DEPLOYMENT.md for Render.com setup
# This gives you an API URL like: https://ccrd-api.onrender.com
```

**Important**: Write down your backend URL. You'll need it for the frontend.

---

## Step 2: Push Code to GitHub

If not already done:

```bash
cd CCRD
git init
git add .
git commit -m "feat: production-ready fraud detection system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CCRD.git
git push -u origin main
```

---

## Step 3: Create Frontend App for Vercel

The frontend needs to be a standalone project. Create a minimal Next.js app:

```bash
# In CCRD root directory
npx create-next-app@latest frontend-app --typescript --tailwind

# Or keep current HTML files and create minimal wrapper
```

**Option: Use Current HTML Files**

Create a simple `pages/index.js` that serves your static files:

```javascript
// frontend-app/pages/index.js
import fs from 'fs';
import path from 'path';

export default function Home() {
  return (
    <div dangerouslySetInnerHTML={{
      __html: fs.readFileSync(
        path.join(process.cwd(), '../frontend/index.html'),
        'utf-8'
      )
    }} />
  );
}
```

**Easier**: Copy HTML files to `public/` folder in a Next.js app.

---

## Step 4: Configure Frontend for API

Update your frontend to use the backend API:

**frontend/config.js** (Updated):
```javascript
// Use environment variable for API URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const token = localStorage.getItem('access_token');
if (!token && window.location.pathname !== "/login.html") {
    window.location.href = 'login.html';
}
```

**Or create .env.local**:
```
NEXT_PUBLIC_API_URL=https://ccrd-api.onrender.com
```

---

## Step 5: Connect Vercel to GitHub

### Method 1: Via Vercel Dashboard (Easiest)

1. Go to [vercel.com](https://vercel.com)
2. Sign in with GitHub
3. Click "New Project"
4. Select your CCRD repository
5. Configure:
   - **Root Directory**: `frontend` (if using separate folder)
   - **Build Command**: `npm run build` (or `next build`)
   - **Output Directory**: `.next` (or `out`)
6. Click "Deploy"

### Method 2: Via Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from your repository
cd frontend
vercel

# Follow prompts to connect GitHub
```

---

## Step 6: Set Environment Variables

In Vercel Dashboard:

1. Go to **Settings** → **Environment Variables**
2. Add:
   ```
   NEXT_PUBLIC_API_URL = https://ccrd-api.onrender.com
   DATABASE_URL = postgresql://... (if needed)
   ```

3. Click **Redeploy** to rebuild with new variables

---

## Step 7: Configure Custom Domain (Optional)

1. In Vercel dashboard: **Settings** → **Domains**
2. Add your custom domain
3. Update DNS records as Vercel instructs
4. Wait for DNS propagation (5-30 minutes)

---

## Step 8: Update CORS on Backend

Update your backend to allow Vercel domain:

**backend/.env**:
```
FRONTEND_URL=https://your-vercel-domain.vercel.app
```

**Or hardcode in** `backend/app/core/config.py`:
```python
cors_origins: list = [
    "https://your-vercel-domain.vercel.app",
    "http://localhost:3000"  # Keep for local dev
]
```

Then redeploy backend.

---

## Testing

### Test Local
```bash
cd frontend
npm run dev
# Visit http://localhost:3000
```

### Test on Vercel
1. Visit your Vercel deployment URL
2. Try signup: Should work if backend is running
3. Try login: Should get JWT token
4. Try prediction: Should hit backend API

---

## Troubleshooting

### "Cannot reach backend API"

```javascript
// Check if API URL is correct
console.log('API URL:', process.env.NEXT_PUBLIC_API_URL);

// Check CORS errors in browser console
// If CORS error, update FRONTEND_URL on backend
```

### "Build fails on Vercel"

1. Check build logs: Vercel Dashboard → **Deployments** → **Build**
2. Common issues:
   - Missing `package.json` in frontend directory
   - Node version incompatibility
   - Missing build command in `vercel.json`

### "Static files not loading"

If using Next.js:
```javascript
// Place static files in public/
public/
├── style.css
├── config.js
└── images/
```

---

## Production Checklist

Before going live:

- [ ] Backend deployed and tested
- [ ] Frontend connected to backend API URL
- [ ] Environment variables set on Vercel
- [ ] CORS configured correctly
- [ ] Custom domain DNS configured (if using)
- [ ] SSL certificate auto-enabled (Vercel does this)
- [ ] All API endpoints return 200s
- [ ] Error handling works
- [ ] Mobile responsive design tested
- [ ] Login/logout flows tested

---

## Monitoring & Logs

### View Vercel Logs
```bash
# Via Vercel CLI
vercel logs

# Or in dashboard: Deployments → Runtime Logs
```

### View Backend Logs
```bash
# If on Render.com
# Go to Render dashboard → Services → Logs
```

---

## Costs

- **Vercel Frontend**: FREE (includes 1 production deployment)
- **Render Backend**: FREE tier (with limitations), $7+/month for production
- **Custom Domain**: ~$12/year
- **Database**: FREE tier on Render (SQLite) or ~$15/month (PostgreSQL)

**Total estimated**: $0-30/month

---

## Next: Advanced Setup

### Add Analytics
```bash
# Vercel includes Web Analytics for free
# Enable in Vercel dashboard → Analytics
```

### Add Performance Monitoring
```bash
# Use Sentry for error tracking
npm install @sentry/nextjs
```

### Add CI/CD to Frontend
```yaml
# .github/workflows/frontend-deploy.yml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: vercel/action@main
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

---

## Support

- **Vercel Docs**: https://vercel.com/docs
- **Next.js Docs**: https://nextjs.org/docs
- **Issues**: Check GitHub Issues or Vercel Status

---

**Your CCRD frontend is now live on Vercel! 🎉**
